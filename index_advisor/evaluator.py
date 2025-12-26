import gc
import multiprocessing as mp
import os
import random
import time
from argparse import Namespace
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Iterable, Sequence, cast

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig

from index_advisor.cache import build_cache
from index_advisor.dataset import init_schema_column_stats, init_workload_predicate_stats
from index_advisor.db import SizeCache, get_hypo_index_size
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model
from lmf_hooks.model import hook_tokenizer, mm_init
from scripts.sft_projector import CustomDataset, SimpleMMDatasetConverter, new_collator_rl

from . import compute_reward
from .db import CostCache, DBOption, create_hypo_index_and_get_cost, create_index_and_get_cost
from .device import set_env_devices
from .eval import IndexRes, ModelResult, Result
from .ia_logging import logger
from .mm.model import MMConfig
from .schemas import Index, parse_indexes_str, schemas
from .utils import compute_workloads_summary
from .workload import load_workloads


logger = logger.getChild("evaluator")
IDX_GUARD = Index("guard", [])  # 用于占位


class Evaluator:
    def __init__(
        self,
        args: Namespace,
        db_option: DBOption | None,
        workload_file: str,
        round: int,
        checkpoints: Sequence[Path | None],
        real: bool,
        is_rank0: bool,
        group_size: int,
    ):
        self.args = args
        self.mm_config = MMConfig.from_args(self.args)
        mm_init(self.mm_config)
        self.converter = SimpleMMDatasetConverter(self.mm_config)  # 由于 embedding 全程不变, 只需初始化一次
        self.db_option = db_option
        self.real = real
        self.round = round
        self.group_size = group_size

        # parse args
        self.model_args, self.data_args, self.finetuning_args, self.generating_args = get_infer_args()

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)
        self.tokenizer = hook_tokenizer(tokenizer)
        assert isinstance(self.tokenizer.pad_token_id, int)
        self.tokenizer.padding_side = "left"
        # dataset
        random.seed(0)
        workloads = load_workloads(workload_file)
        if args.eval_set_percent > 0:
            eval_size = int(len(workloads) * args.eval_set_percent)
            train_size = len(workloads) - eval_size
            logger.info(f"拆分数据集，训练集大小: {train_size}, 评估集大小: {eval_size}")
            random.shuffle(workloads)
            train_set = workloads[:train_size]
            eval_set = workloads[train_size:]
            # 为 eval_set 创建摘要以跨训练验证其一致性. 确保当 dataset 和 eval_set_percent 相同时，eval_dataset 在不同运行中保持不变
            eval_summary = compute_workloads_summary(eval_set)
            logger.info(f"评估数据集摘要: {eval_summary}")
        else:
            train_set = load_workloads(workload_file)
            eval_set = None

        if self.args.ptype == "prompt1":
            assert self.db_option is not None
            logger.info("启用 to_prompt1, 预计算列统计信息和谓词信息")
            init_schema_column_stats(self.db_option, workers=16)
            stats_targets = list(train_set)
            if eval_set:
                stats_targets += eval_set
            logger.info(f"预计算谓词统计信息，数据集大小: {len(stats_targets)}")
            init_workload_predicate_stats(stats_targets, self.db_option, workers=16)

        self.cost_cache = CostCache()
        self.size_cache = SizeCache()
        dataset = CustomDataset(
            train_set,
            tokenizer,
            self.mm_config,
            db_config_d=db_option.to_dict() if db_option else None,
            generate=True,
            extend=False,
            cost_cache=self.cost_cache,
            size_cache=self.size_cache,
            group_size=self.group_size,
            prompt_type=self.args.ptype,
        )
        self.workloads = dataset.data
        self.dataset = dataset

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,  # type: ignore
            **self.generating_args.to_dict(),
        )
        # 获取所有checkpoint路径
        self.checkpoint_dirs = checkpoints

    def _load_checkpoint(self, checkpoint: Path | None):
        """加载检查点"""
        if checkpoint is not None:
            if os.path.exists(checkpoint / "adapter_config.json"):
                self.model_args.adapter_name_or_path = [str(checkpoint)]  # type: ignore
                logger.info(f"设置 checkpoint: {str(checkpoint)}")
            else:
                self.model_args.adapter_name_or_path = None
                logger.warning("设置了 checkpoint 但未找到 adapter_config.json, 仅加载 projector")
            if not self.args.mm_projector_checkpoint and self.mm_config.enable:
                self.mm_config.projector_checkpoint = str(checkpoint / "projector.safetensors")
                logger.info(f"设置 column projector: {self.mm_config.projector_checkpoint}")
            if not self.args.mm2_projector_checkpoint and self.mm_config.enable_sql:
                self.mm_config.projector2_checkpoint = str(checkpoint / "projector.safetensors")
                logger.info(f"设置 sql projector: {self.mm_config.projector2_checkpoint}")
        if hasattr(self, "model"):
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
        self.model = load_model(self.tokenizer, self.model_args, self.finetuning_args)
        self.model.to("cuda:0")  # type: ignore

    def evaluate(self, batch_size: int) -> list[ModelResult]:
        """评估所有checkpoint"""
        collator = new_collator_rl(self.tokenizer.pad_token_id)  # type: ignore
        all_results = []

        for cp in tqdm(self.checkpoint_dirs, desc="Checkpoints", unit="ckpt"):
            if cp is not None:
                model_name = cp.name
            elif (p := self.model_args.model_name_or_path) and isinstance(p, list):
                model_name = Path(p[0]).name
            else:
                model_name = "unknown"
            # 重新加载模型
            self._load_checkpoint(cp)
            ckpt_res: list[Result] = []

            # 清空 labels, 后续将使用 labels 存储每一步的推荐结果
            for w in self.workloads:
                w.labels = [IDX_GUARD] * self.group_size

            dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=4,
                multiprocessing_context=mp.get_context("fork"),
            )

            logger.info(f"开始评估模型 {model_name}, 推理轮数: {self.round}")
            for turn in range(1, self.round + 1):
                logger.info(f"🔄 开始第 {turn} 轮")
                logger.info(f"{len(self.cost_cache)=} {len(self.size_cache)=}")
                gc.collect()
                torch.cuda.empty_cache()

                for batch_idx, batch in tqdm(
                    enumerate(dataloader), desc=f"{model_name} - turn {turn}", unit="batch", total=len(dataloader)
                ):
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(self.model.device)

                    with torch.no_grad():
                        outputs = self.model.generate(generation_config=self.generation_config, **batch)

                    batch = cast("dict[str, torch.Tensor]", batch)
                    query = batch["input_ids"].detach().cpu()
                    response = outputs[:, query.size(-1) :].detach().cpu()
                    for i in range(len(query)):
                        w = self.workloads[batch_idx * batch_size + i]
                        w.labels = cast("list[Index]", w.labels)
                        remaining = self.round * self.group_size - len(w.labels) + self.group_size
                        if remaining <= 0:
                            continue

                        expected = min(self.group_size, remaining)

                        t1 = cast(torch.Tensor, query[i] != self.tokenizer.pad_token_id)
                        query_start_index = t1.nonzero()[0].item()
                        t2 = cast(torch.Tensor, response[i] != self.tokenizer.pad_token_id)
                        response_indexes = t2.nonzero()

                        if len(response_indexes) == 0:
                            response_length = 1
                        elif self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:
                            response_length = response_indexes[-1].item() + 2
                        else:
                            response_length = response_indexes[-1].item() + 1

                        q = self.tokenizer.decode(query[i, query_start_index:], True, True)
                        r = self.tokenizer.decode(response[i, :response_length], True, True)
                        logger.debug(f"💬 Workload {w.id} 第 {turn} 轮对话\nQ: {(q,)}\nR: {r}")
                        indexes, (invalid, exist, duplicate) = parse_indexes_str(r, schemas[w.db], True)
                        if len(indexes) > expected:
                            logger.debug(f"⚠️ 解析出 {len(indexes)} 个索引, 保留前 {expected} 个")
                        elif len(indexes) < expected:
                            logger.warning(f"🏷️ Workload {w.id} 第 {turn} 轮推荐索引少于{expected}")
                            # 补齐占位符
                            indexes.extend([IDX_GUARD] * (expected - len(indexes)))

                        w.labels = w.labels[: -self.group_size] + indexes[:expected]
                        logger.info(f"🏷️ Workload {w.id} 第 {turn} 轮推荐索引: {indexes[:expected]}")
                        w.labels.extend([IDX_GUARD] * self.group_size)

            for w in self.workloads:
                # 移除多余的占位符
                w.labels = [idx for idx in (w.labels or []) if idx != IDX_GUARD]

            if self.db_option is None:
                logger.info("⚠️ 未提供数据库连接信息, 跳过评估")
                for w in self.workloads:
                    ckpt_res.append(
                        Result(
                            workload_id=w.id,
                            db=w.db,
                            queries=w.sqls,
                            sqls_hash=w.sqls_hash,
                            indexes=[],
                            basic_cost_explain=-1,
                            basic_cost_real=None,
                            actual_cost_hypo=-1,
                            actual_cost_real=None,
                            reward=0.0,
                            prompt="",
                            response="\n".join(str(idx) for idx in (w.labels or [])),
                            invalid=0,
                            exist=0,
                            duplicate=0,
                            useless=0,
                        )
                    )
                all_results.append(ModelResult(name=model_name, results=ckpt_res))
                continue

            logger.info("构建成本和大小缓存...")
            t = time.time()
            self.cost_cache, self.size_cache = build_cache(
                self.workloads,
                self.db_option,
                16,
                self.cost_cache,
                self.size_cache,
            )
            logger.info(f"构建缓存完成, 用时 {time.time() - t:.2f} 秒")

            # 评估推荐结果
            for w in tqdm(self.workloads, desc="Calculating Rewards", unit="workload"):
                conn = self.db_option.connections[w.db]
                w.basic_cost_explain = create_hypo_index_and_get_cost(conn, [], w.queries, self.cost_cache)
                logger.debug(f"🏷️ 无索引 cost: {w.basic_cost_explain}")

                index_res: list[IndexRes] = []
                indexes = w.labels or []
                res = []
                total_size = 0.0
                prev_cost = w.basic_cost_explain
                for idx in indexes:
                    size = get_hypo_index_size(conn.cursor(), w.db, idx, self.size_cache)
                    total_size += size
                    res.append(idx)
                    cur_cost = create_hypo_index_and_get_cost(conn, res, w.queries, self.cost_cache)
                    profit = prev_cost - cur_cost
                    prev_cost = cur_cost
                    index_res.append(IndexRes(str(idx), 0, size=size, total_size=total_size, profit=profit))

                actual_cost_hypo = create_hypo_index_and_get_cost(conn, res, w.queries, self.cost_cache)
                logger.debug(f"🏷️ 假设索引 cost: {actual_cost_hypo}")
                total_size /= 1024 * 1024
                logger.info(f"🏷️ 索引总大小: {total_size}MB")

                actual_cost_real = None
                if self.real:
                    if w.basic_cost_real is None:
                        w.basic_cost_real = create_index_and_get_cost(conn, [], w.queries, True, self.cost_cache)
                    logger.debug(f"🏷️ 无索引真实 cost: {w.basic_cost_real}")
                    actual_cost_real = create_index_and_get_cost(conn, res, w.queries, True, self.cost_cache)
                    logger.debug(f"🏷️ 真实索引 cost: {actual_cost_real}")

                reward = compute_reward(
                    w.basic_cost,
                    actual_cost_real or actual_cost_hypo,
                    total=1,
                    invalid=0,
                    exist=0,
                    duplicate=0,
                )
                if total_size != 0:
                    reward /= total_size
                else:
                    logger.warning("⚠️ 索引大小为 0, 无法计算 reward")
                    reward = 0
                logger.debug(f"🎯 计算 reward: {reward}")
                ckpt_res.append(
                    Result(
                        workload_id=w.id,
                        db=w.db,
                        queries=w.sqls,
                        sqls_hash=w.sqls_hash,
                        indexes=index_res,
                        basic_cost_explain=w.basic_cost_explain,
                        basic_cost_real=w.basic_cost_real,
                        actual_cost_hypo=actual_cost_hypo,
                        actual_cost_real=actual_cost_real,
                        reward=reward,
                        prompt="",
                        response="\n".join(str(idx) for idx in res),
                        invalid=0,
                        exist=0,
                        duplicate=0,
                        useless=0,
                    )
                )

            all_results.append(ModelResult(name=model_name, results=ckpt_res))

        return all_results


def run_on_device(
    args,
    db_option_dict: dict | None,
    workload_file: str,
    round: int,
    checkpoints: Sequence[Path | None],
    real: bool,
    device_id: int,
    is_rank0: bool,
    batch_size: int,
    group_size: int,
    result_queue: "mp.Queue[tuple[int, list[ModelResult] | None]]",
):
    try:
        evaluator = Evaluator(
            args,
            DBOption.from_dict(db_option_dict) if db_option_dict else None,
            workload_file,
            round,
            checkpoints,
            real,
            is_rank0,
            group_size,
        )
        results = evaluator.evaluate(batch_size)
        logger.debug(f"设备 {device_id} evaluator.evaluate 完成")
        result_queue.put((device_id, results))
        logger.info(f"设备 {device_id} results 发送完成")
    except Exception as e:
        logger.error(f"设备 {device_id} 评估失败: {str(e)}", exc_info=True)
        result_queue.put((device_id, None))


class ParallelEvaluator:
    def __init__(
        self,
        args: Namespace,
        db_option: DBOption | None,
        workload_file: str,
        round: int,
        ckpt_paths: Iterable[Path] | None,
        real: bool,
        device_ids: list[int],
        group_size: int,
    ):
        self.args = args
        self.real = real
        self.device_ids = device_ids
        self.round = round
        self.group_size = group_size
        self.workload_file = workload_file

        # 获取所有checkpoint路径
        self.model_args, self.data_args, self.finetuning_args, self.generating_args = get_infer_args()
        if args.adapter_dir:
            self.model_args.adapter_name_or_path = [args.adapter_dir]  # type: ignore
        elif self.model_args.adapter_name_or_path:
            raise ValueError("请通过 --adapter-dir 指定 LoRA adapter 路径, 配置文件中的 adapter_name_or_path 将被忽略")
        self.checkpoint_dirs = self._get_checkpoint_dirs(ckpt_paths)

        # 分割任务
        self.rank_ckpts = self._split_tasks()

        # 初始化连接池
        self.db_option = db_option

    def _get_checkpoint_dirs(self, ckpt_paths: Iterable[Path] | None) -> Sequence[Path | None]:
        assert isinstance(self.model_args.adapter_name_or_path, list | None)
        if ckpt_paths is None:
            return [None]  # 评估原始模型
        else:
            return list(ckpt_paths)

    def _split_tasks(self):
        """将检查点分配到不同设备"""
        num_devices = len(self.device_ids)

        # 分割检查点
        rank_ckpts: dict[int, list[Path | None]] = {}
        for i, ckpt in enumerate(self.checkpoint_dirs):
            device_idx = i % num_devices
            rank_ckpts.setdefault(device_idx, []).append(ckpt)

        logger.info("分配 ckeckpoint 到设备:")
        for device_idx, ckpts in rank_ckpts.items():
            logger.info(f"设备 {self.device_ids[device_idx]}: {[p.name if p else '原始模型' for p in ckpts]}")
        return rank_ckpts

    def evaluate(self, batch_size: int = 32) -> list[ModelResult]:
        """并行评估所有设备上的任务"""
        # 创建结果队列和进程
        ctx = mp.get_context("spawn")
        result_queue: mp.Queue[tuple[int, list[ModelResult] | None]] = ctx.Queue()
        processes: list[BaseProcess] = []

        # 启动子进程
        for idx, device_id in enumerate(self.device_ids):
            checkpoints = self.rank_ckpts.get(idx, [])
            if not checkpoints:
                logger.info(f"设备 {device_id} 没有待评估的 checkpoint, 跳过")
                continue
            set_env_devices(os.environ, device_id)
            p = ctx.Process(
                target=run_on_device,
                args=(
                    self.args,
                    self.db_option.to_dict() if self.db_option else None,
                    self.workload_file,
                    self.round,
                    checkpoints,
                    self.real,
                    device_id,
                    len(processes) == 0,
                    batch_size,
                    self.group_size,
                    result_queue,
                ),
            )
            p.start()
            processes.append(p)

        # 收集结果
        all_results = []
        for _ in range(len(processes)):
            device_idx, results = result_queue.get()
            if results is not None:
                all_results.extend(results)
                logger.info(f"设备 {device_idx} 评估完成")
            else:
                logger.error(f"设备 {device_idx} 评估失败, 请查看日志")
                raise RuntimeError(f"设备 {device_idx} 评估失败")

        # 等待所有进程结束
        for p in processes:
            p.join()

        return all_results

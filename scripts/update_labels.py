import argparse
import gc
import multiprocessing as mp
import os
from dataclasses import dataclass
from datetime import datetime
from itertools import repeat
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Iterable, Sequence, cast

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig

from index_advisor import parse_args_safe
from index_advisor.db import DBOption, create_hypo_index_and_get_cost, get_hypo_index_size
from index_advisor.device import set_env_devices
from index_advisor.ia_logging import log_to_file, logger
from index_advisor.mm.model import MMConfig
from index_advisor.schemas import Index, parse_indexes_str, schemas
from index_advisor.utils import parse_ranges
from index_advisor.workload import Workload, load_workloads, save_workloads
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model
from lmf_hooks.model import hook_tokenizer, mm_init
from scripts.sft_projector import CustomDataset, new_collator_rl


LOG = logger.getChild("update_labels")


@dataclass
class Prediction:
    w: Workload
    indexes: list[Index | None]
    # invalid: int
    # exist: int
    # duplicate: int
    # response: str


@dataclass
class CheckpointPredictions:
    name: str
    predictions: list[Prediction]


class LabelInferencer:
    def __init__(
        self,
        args: argparse.Namespace,
        workload_file: str,
        checkpoints: Sequence[Path | None],
        is_rank0: bool,
    ) -> None:
        self.args = args
        self.workload_file = workload_file
        self.checkpoints = checkpoints
        self.is_rank0 = is_rank0
        self.group_size = args.group_size

        self.mm_config = MMConfig.from_args(self.args)
        mm_init(self.mm_config)

        self.model_args, self.data_args, self.finetuning_args, self.generating_args = get_infer_args()
        if args.adapter_dir:
            self.model_args.adapter_name_or_path = [args.adapter_dir]  # type: ignore[attr-defined]

        tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)
        self.tokenizer = hook_tokenizer(tokenizer)
        assert isinstance(self.tokenizer.pad_token_id, int)
        self.tokenizer.padding_side = "left"

        workloads = load_workloads(workload_file)
        self.raw_workloads = {w.id: w for w in workloads}

        self.dataset = CustomDataset(
            workloads,
            self.tokenizer,
            self.mm_config,
            generate=True,
            extend=True,
            log_sample=self.is_rank0,
            group_size=self.group_size,
        )
        self.workloads = self.dataset.data

        eos_ids = [self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids  # type: ignore[attr-defined]
        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_ids,
            **self.generating_args.to_dict(),
        )

    def _load_checkpoint(self, checkpoint: Path | None) -> str:
        if checkpoint is not None:
            self.model_args.adapter_name_or_path = [str(checkpoint)]  # type: ignore[attr-defined]
            if not self.args.mm_projector_checkpoint and self.mm_config.enable:
                self.mm_config.projector_checkpoint = str(checkpoint / "projector.safetensors")
            if not self.args.mm2_projector_checkpoint and self.mm_config.enable_sql:
                self.mm_config.projector2_checkpoint = str(checkpoint / "projector.safetensors")
            model_name = checkpoint.name
        else:
            model_path = self.model_args.model_name_or_path
            if isinstance(model_path, list) and model_path:
                model_name = Path(model_path[0]).name
            elif isinstance(model_path, str):
                model_name = Path(model_path).name
            else:
                model_name = "base"

        if hasattr(self, "model"):
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()

        self.model = load_model(self.tokenizer, self.model_args, self.finetuning_args)
        self.model.to("cuda:0")  # type: ignore[arg-type]
        return model_name

    def infer(self, batch_size: int) -> list[CheckpointPredictions]:
        if batch_size <= 0:
            raise ValueError("batch size 必须大于 0")

        collator = new_collator_rl(self.tokenizer.pad_token_id)
        all_predictions: list[CheckpointPredictions] = []

        for checkpoint in tqdm(self.checkpoints, desc="Checkpoints", unit="ckpt"):
            model_name = self._load_checkpoint(checkpoint)
            predictions: dict[int, Prediction] = {}

            dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=4,
                multiprocessing_context=mp.get_context("fork"),
            )

            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{model_name}", unit="batch", leave=False)):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(generation_config=self.generation_config, **batch)

                input_tensor = cast(torch.Tensor, batch["input_ids"])
                inputs = input_tensor.detach().cpu()
                output_tensor = cast(torch.Tensor, outputs)
                responses = output_tensor[:, inputs.size(-1) :].detach().cpu()

                for i in range(len(inputs)):
                    input_tokens = inputs[i]
                    response_tokens = responses[i]

                    non_pad = (input_tokens != self.tokenizer.pad_token_id).nonzero()
                    if len(non_pad) == 0:
                        continue

                    response_non_pad = (response_tokens != self.tokenizer.pad_token_id).nonzero()
                    if len(response_non_pad) == 0:
                        response_len = 1
                    elif self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:
                        response_len = int(response_non_pad[-1]) + 2
                    else:
                        response_len = int(response_non_pad[-1]) + 1

                    decoded = self.tokenizer.decode(response_tokens[:response_len], skip_special_tokens=True)

                    extended = self.workloads[batch_idx * batch_size + i]  # 无完整 labels 字段
                    w = self.raw_workloads[extended.id]  # 有完整 labels 字段
                    label_count = len(w.labels or [])
                    pred_entry = predictions.setdefault(w.id, Prediction(w, indexes=[None] * label_count))

                    schema = schemas[w.db]
                    indexes, (invalid, exist, duplicate) = parse_indexes_str(decoded, schema, True)

                    if len(indexes) < self.group_size:
                        LOG.debug(
                            f"workload {w.id} checkpoint {model_name} 解析索引数量 {len(indexes)} < {self.group_size}, 跳过"
                        )
                        continue

                    if len(indexes) > self.group_size:
                        LOG.debug(
                            f"workload {w.id} checkpoint {model_name} 解析出 {len(indexes)} 个索引, 仅保留前 {self.group_size} 个"
                        )

                    start_pos = label_count - self.group_size
                    pred_entry.indexes[start_pos : start_pos + self.group_size : 1] = indexes[: self.group_size]

            all_predictions.append(CheckpointPredictions(name=model_name, predictions=list(predictions.values())))

        return all_predictions


def run_label_inference_on_device(
    args: argparse.Namespace,
    workload_file: str,
    checkpoints: Sequence[Path | None],
    device_id: int,
    is_rank0: bool,
    batch_size: int,
    result_queue: "mp.Queue[tuple[int, list[CheckpointPredictions] | None]]",
) -> None:
    try:
        inferencer = LabelInferencer(args, workload_file, checkpoints, is_rank0)
        results = inferencer.infer(batch_size)
        result_queue.put((device_id, results))
    except Exception as exc:
        LOG.error(f"设备 {device_id} 推理失败: {exc}", exc_info=True)
        result_queue.put((device_id, None))


class ParallelLabelInferencer:
    def __init__(
        self,
        args: argparse.Namespace,
        workload_file: str,
        ckpt_paths: Iterable[Path] | None,
        device_ids: list[int],
    ) -> None:
        self.args = args
        self.workload_file = workload_file
        self.device_ids = device_ids

        self.model_args, self.data_args, self.finetuning_args, self.generating_args = get_infer_args()
        if args.adapter_dir:
            self.model_args.adapter_name_or_path = [args.adapter_dir]  # type: ignore[attr-defined]
        elif self.model_args.adapter_name_or_path:
            raise ValueError("请通过 --adapter-dir 指定 LoRA adapter 路径, 配置文件中的 adapter_name_or_path 将被忽略")

        self.checkpoint_dirs = self._get_checkpoint_dirs(ckpt_paths)
        self.rank_ckpts = self._split_tasks()

    def _get_checkpoint_dirs(self, ckpt_paths: Iterable[Path] | None) -> Sequence[Path | None]:
        assert isinstance(self.model_args.adapter_name_or_path, list | None)
        if ckpt_paths is None:
            return [None]  # 评估原始模型
        else:
            return list(ckpt_paths)

    def _split_tasks(self) -> dict[int, list[Path | None]]:
        rank_ckpts: dict[int, list[Path | None]] = {}
        if not self.checkpoint_dirs:
            return rank_ckpts
        device_count = max(1, len(self.device_ids))
        for idx, ckpt in enumerate(self.checkpoint_dirs):
            device_idx = idx % device_count
            rank_ckpts.setdefault(device_idx, []).append(ckpt)

        LOG.info("分配 ckeckpoint 到设备:")
        for device_idx, ckpts in rank_ckpts.items():
            LOG.info(f"设备 {self.device_ids[device_idx]}: {[p.name if p else '原始模型' for p in ckpts]}")

        return rank_ckpts

    def infer(self, batch_size: int) -> list[CheckpointPredictions]:
        ctx = mp.get_context("spawn")
        result_queue: mp.Queue[tuple[int, list[CheckpointPredictions] | None]] = ctx.Queue()
        processes: list[BaseProcess] = []

        for idx, device_id in enumerate(self.device_ids):
            checkpoints = self.rank_ckpts.get(idx, [])
            if not checkpoints:
                LOG.info(f"设备 {device_id} 无待推理 checkpoint, 跳过")
                continue
            set_env_devices(os.environ, device_id)
            process = ctx.Process(
                target=run_label_inference_on_device,
                args=(
                    self.args,
                    self.workload_file,
                    checkpoints,
                    device_id,
                    len(processes) == 0,
                    batch_size,
                    result_queue,
                ),
            )
            process.start()
            processes.append(process)

        res = []
        for _ in range(len(processes)):
            device_idx, results = result_queue.get()
            if results is None:
                LOG.error(f"设备 {device_idx} 推理失败, 请检查日志")
                continue
            res.extend(results)
            LOG.info(f"设备 {device_idx} 推理完成")

        for process in processes:
            process.join()

        return res


@dataclass
class Candidate:
    source: str
    prediction: Prediction


def update_labels(predictions: list[CheckpointPredictions], db_option: DBOption, workers: int, group_size: int = 1):
    # 1. CheckpointPredictions 的封装层次是 ckpt -> workload_id -> Prediction
    # 2. 我们需要将其转换为 workload_id -> Candidates(ckpt 来源 + Prediction)
    workload_candidates: dict[int, list[Candidate]] = {}
    add_raw = True
    for ckpt_preds in predictions:
        for pred in ckpt_preds.predictions:
            workload_candidates.setdefault(pred.w.id, []).append(Candidate(source=ckpt_preds.name, prediction=pred))
            if add_raw:
                raw_labels = Prediction(pred.w, cast(list[Index | None], pred.w.effective_labels()))
                workload_candidates.setdefault(pred.w.id, []).append(Candidate(source="base", prediction=raw_labels))
        add_raw = False  # 只添加一次原始标签

    with mp.get_context("spawn").Pool(workers) as pool:
        db_option_d = db_option.to_dict()
        res = pool.starmap(
            update_labels_worker, zip(workload_candidates.items(), repeat(db_option_d), repeat(group_size))
        )

    workloads = [w for w, _, _ in res]
    updated = sum(u for _, u, _ in res)
    total = sum(t for _, _, t in res)
    return workloads, updated, total


def update_labels_worker(workload_candidate: tuple[int, list[Candidate]], db_option_d: dict, group_size: int):
    db_option = DBOption.from_dict(db_option_d)
    workload_id, candidates = workload_candidate
    w = candidates[0].prediction.w
    labels = w.labels or []

    def max_candidates(candidates: list[Candidate]) -> list[tuple[Candidate, float]]:
        """计算每个 group_size 区段最好的 Candidate 及其 profit."""
        conn = db_option.connections[w.db]
        size_cursor = conn.cursor()

        def max_candidate_for_group(start: int):
            """计算某个 candidate 的 [start, start + group_size) 区段的 profit."""
            prefix_indexes = labels[:start]
            prefix_cost = create_hypo_index_and_get_cost(conn, prefix_indexes, w.queries)

            def candidate_profit(c: Candidate) -> float:
                indexes = c.prediction.indexes[start : start + group_size]
                if any(idx is None for idx in indexes):
                    LOG.debug(f"workload {w.id} {c.source} {start}-{start + group_size - 1} 存在 None, 跳过")
                    return 0.0
                indexes = cast(list[Index], indexes)
                improvement = prefix_cost - create_hypo_index_and_get_cost(conn, prefix_indexes + indexes, w.queries)
                index_size = sum(get_hypo_index_size(size_cursor, w.db, idx) for idx in indexes) / (1024 * 1024)
                if prefix_cost == 0 or index_size == 0:
                    LOG.warning(
                        f"workload {w.id} {c.source} {start}-{start + group_size - 1} 索引: {indexes} {prefix_cost=} {index_size=} 跳过"
                    )
                    return 0.0
                profit = improvement / prefix_cost / index_size
                LOG.info(
                    f"workload {w.id} {c.source} {start}-{start + group_size - 1} 索引: {indexes} {improvement=} {index_size=} {profit=}"
                )
                return profit

            return candidate_profit

        group_count = len(labels) // group_size
        result: list[tuple[Candidate, float]] = []
        for i in range(group_count):
            start = i * group_size
            profits = [(c, max_candidate_for_group(start)(c)) for c in candidates]
            best = max(profits, key=lambda item: item[1])
            result.append(best)
        return result

    def format_update(old: list[Index], new: list[Index]) -> str:
        return ", ".join(f"{o} -> {n}" for o, n in zip(old, new))

    updated = 0
    group_bests = max_candidates(candidates)
    for i, (b, profit) in enumerate(group_bests):
        start = i * group_size
        end = start + group_size
        best = b.prediction.indexes[start:end]
        if profit <= 0:
            LOG.info(f"workload {workload_id} {start}-{end - 1} 所有候选 profit<=0, 跳过更新")
            continue
        assert all(idx is not None for idx in best)
        best = cast(list[Index], best)

        raw = labels[start:end]
        assert len(raw) == len(best) == group_size, f"{len(raw)=} {len(best)=} {group_size=} 不相等"

        if b.source == "base" or set(best) == set(raw):
            continue

        # todo 对 best 排序后再更新
        for j in range(group_size):
            w.label_overrides[start + j] = best[j]
        LOG.info(f"workload {workload_id} {start}-{end - 1} 标签更新: {format_update(raw, best)} (from {b.source})")
        updated += 1
    return w, updated, len(group_bests)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="进行单步推理推理并更新 workload 标签", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("workload_file", help="输入 workload 文件路径")
    parser.add_argument("-o", "--output", help="输出 workload 文件路径. 默认为输入路径附加时间戳", default=None)
    parser.add_argument("--desc", help="任务描述", default="更新 workload labels")
    parser.add_argument("--ckpt", default="all")
    parser.add_argument("--gpu", help="要使用的 GPU, 同 --ckpt 格式", default="0")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="推理 batch size")
    parser.add_argument("--group-size", "--gs", type=int, default=1, help="每轮推荐的索引数量")
    parser.add_argument("-r", "--real", action="store_true", help="使用真实索引执行评估 (EXPLAIN ANALYZE)")
    parser.add_argument("--eval-workers", type=int, default=16, help="评估候选索引时使用的并行进程数")
    parser.add_argument("--adapter-dir", help="LoRA adapter 路径, 将覆盖配置中的 adapter_name_or_path", default=None)
    parser.add_argument("--debug", action="store_true", help="调试模式")

    DBOption.add_arg_group_to(parser)
    MMConfig.add_arg_group_to(parser)

    args = parse_args_safe(parser)

    if args.batch_size <= 0:
        raise ValueError("batch size 必须大于 0")
    if args.group_size <= 0:
        raise ValueError("group size 必须大于 0")

    # parse checkpoints
    if args.adapter_dir is None:
        ckpts = None
    elif args.ckpt == "all":
        ckpts = Path(args.adapter_dir).parent.glob("checkpoint-*")
    elif not args.ckpt:
        raise ValueError("指定 --adapter-dir 同时必须指定 --ckpt")
    else:
        ckpts = [Path(args.adapter_dir).parent / f"checkpoint-{s}" for s in parse_ranges(args.ckpt)]
        ckpts = [p for p in ckpts if p.is_dir()]

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%y%m%d%H%M%S")
        workload_path = Path(args.workload_file)
        output_path = workload_path.with_name(f"{workload_path.stem}.updated-{timestamp}{workload_path.suffix}")

    log_to_file(
        task="update_labels", level="DEBUG" if args.debug else "INFO", desc=args.desc, log_dir=output_path.parent
    )

    logger.info(f"使用检查点: {ckpts}")

    # parse devices
    device_ids = parse_ranges(args.gpu)
    logger.info(f"使用 gpu: {device_ids}")

    db_option = DBOption.from_args(args)

    if args.adapter_dir:
        args.adapter_dir = os.path.abspath(args.adapter_dir)

    inferencer = ParallelLabelInferencer(
        args=args,
        workload_file=args.workload_file,
        ckpt_paths=ckpts,
        device_ids=device_ids,
    )

    predictions = inferencer.infer(args.batch_size)

    workloads, updated, total = update_labels(
        predictions,
        db_option,
        workers=args.eval_workers,
        group_size=args.group_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_workloads(output_path, workloads)

    LOG.info(f"更新完成: {updated}/{total} 个标签组被更新")
    LOG.info(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()

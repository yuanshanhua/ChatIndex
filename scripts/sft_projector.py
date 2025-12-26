import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from index_advisor import parse_args_safe
from index_advisor.dataset import init_schema_column_stats, init_workload_predicate_stats
from index_advisor.db import Connection, CostCache, DBOption, SizeCache
from index_advisor.ia_logging import log_to_file, logger
from index_advisor.mm.model import DATASET_COLUMNS, DATASET_SQLS, MMConfig
from index_advisor.msg import continue_prompt, to_data_sample
from index_advisor.schemas import Index
from index_advisor.utils import compute_workloads_summary, resolve_ckpt
from index_advisor.workload import Workload, load_workloads
from llamafactory.hparams import (
    DataArguments,
    FinetuningArguments,
    GeneratingArguments,
    ModelArguments,
    TrainingArguments,
    get_train_args,
)
from llamafactory.model import load_model
from lmf_hooks.model import hook_sft_model, hook_tokenizer, mm_init


if TYPE_CHECKING:
    from torch import Tensor

try:
    # 适配 Ascend npu, 不要删除
    from torch_npu.contrib import transfer_to_npu  # type: ignore
except ImportError:
    pass

logger = logger.getChild("sft_projector")
IDX_GUARD = Index("guard", [])


class SimpleMMDatasetConverter:
    """简化的多模态数据集转换器，不依赖 LLaMA-Factory"""

    def __init__(self, mm_config: MMConfig) -> None:
        self.mm_config = mm_config

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """转换数据样本"""
        result = example.copy()

        # 检查样本的数据完整性
        self._validate_example(example)

        # 处理 SQL token
        if not self.mm_config.enable_sql:
            result.pop(DATASET_SQLS, None)
        elif DATASET_SQLS in example and example[DATASET_SQLS]:
            # 添加 SQL 嵌入到 DATASET_SQLS 字段
            sql_embeddings = []
            for sql_text in example[DATASET_SQLS]:
                try:
                    embedding = self.mm_config.get_sql_embedding(sql_text)
                    sql_embeddings.append(embedding)
                except ValueError as e:
                    raise ValueError(f"SQL '{sql_text}' 的嵌入不存在，数据不完整: {e}")

            if sql_embeddings:
                result[DATASET_SQLS] = torch.stack(sql_embeddings).to(dtype=self.mm_config.projector2_dtype)

        # 处理 Column token
        if not self.mm_config.enable:
            result.pop(DATASET_COLUMNS, None)
        elif DATASET_COLUMNS in example and example[DATASET_COLUMNS]:
            column_embeddings = []
            for column_name in example[DATASET_COLUMNS]:
                if column_name in self.mm_config.column_embeddings:
                    embedding = self.mm_config.column_embeddings[column_name]
                    column_embeddings.append(embedding)
                else:
                    raise ValueError(f"列 '{column_name}' 的嵌入不存在，数据不完整")

            if column_embeddings:
                result[DATASET_COLUMNS] = torch.stack(column_embeddings).to(dtype=self.mm_config.projector_dtype)

        return result

    def _validate_example(self, example: dict[str, Any]) -> None:
        """验证单个数据样本的完整性"""
        # 如果启用了列投影器训练，检查列数据
        if self.mm_config.projector_trainable:
            if DATASET_COLUMNS not in example:
                raise ValueError(f"启用了列投影器训练但数据样本缺少 '{DATASET_COLUMNS}' 字段")

        # 如果启用了SQL投影器训练，检查SQL数据
        if self.mm_config.projector2_trainable:
            if DATASET_SQLS not in example:
                raise ValueError(f"启用了SQL投影器训练但数据样本缺少 '{DATASET_SQLS}' 字段")


def expand_workloads(workloads: list[Workload], *, extend: bool, generate: bool, group_size: int) -> list[Workload]:
    """按需扩展 workloads 以构造多轮对话样本"""

    expanded: list[Workload] = []
    for workload in workloads:
        # 用于 sft(generate=False) 且 labels 不足则报错
        if not generate and (workload.labels is None or len(workload.labels) < group_size):
            raise ValueError(f"SFT 训练样本 labels 不足: {workload.id}")
        labels = workload.labels
        # 无须展开时, 或 labels 不足时, 仅添加原始 workload
        if not extend or labels is None or len(labels) < group_size:
            expanded.append(workload)
            continue

        for end in range(group_size, len(labels) + 1, group_size):
            targets = [workload.label_at(i) for i in range(end - group_size, end)]
            expanded.append(workload.with_labels(labels[: end - group_size] + targets))

    return expanded


class CustomDataset(Dataset):
    def __init__(
        self,
        workloads: list[Workload],
        tokenizer: PreTrainedTokenizer,
        mm_config: MMConfig,
        db_config_d: Optional[dict[str, Any]] = None,
        *,
        multi_round: bool = True,
        generate: bool = False,
        extend: bool = True,
        log_sample: bool = False,
        cost_cache: CostCache = CostCache(),
        size_cache: SizeCache = SizeCache(),
        group_size: int = 1,
        max_queries: Optional[int] = None,
        prompt_type: Literal["general", "prompt1", "plus"] = "general",
    ) -> None:
        """
        用于 SFT, RL rollout, evaluate 的数据集.

        (extend, generate) 组合含义:
        - (True, True): 用于 RL rollout 或 update labels, 进行单步生成.
        - (True, False): 用于 SFT, 将 workloads 按 labels 扩展为多轮对话样本.
        - (False, True): 用于 evaluate, 不使用标签.
        - (False, False): 无意义.

        Args:
            db_config_d: db_config 序列化所得 dict, 因为 DBOption 不能被 pickle, 需要手动反序列化以支持多进程. 用于生成多轮对话样本.
            generate: 用于 RL rollout 或 evaluate 时需为 True, 否则用于 SFT.
            extend: 将传入的 workloads 按 labels 扩展为多轮对话样本. 用于 evaluate 时不启用, 其他情况按需启用.
            log_sample: 是否打印第一个样本到日志, 一般仅主进程启用.
        """
        # if multi_round:
        #     assert db_config_d is not None, "multi_round 模式下需要提供 db_config_d 以生成多轮对话样本"
        self.tokenizer = tokenizer
        self.mm_config = mm_config
        self.db_config_d = db_config_d
        self.db_config: DBOption | None = None
        self.connections: dict[str, Connection] = {}
        self.multi_round = multi_round
        self.generate = generate
        self.extend = extend
        self._logged = not log_sample
        self.group_size = group_size

        self.cost_cache = cost_cache
        self.size_cache = size_cache
        self.prompt_type = prompt_type

        self.converter = SimpleMMDatasetConverter(mm_config)
        self.workloads = workloads

        if self.extend and not self.generate:
            # invalid = [w.id for w in workloads if not w.labels or len(w.labels) < self.group_size]
            # if invalid:
            #     raise ValueError(f"以下 workload 标签少于 {self.group_size}, 无法用于 SFT: {sorted(invalid)}")
            raw = len(workloads)
            workloads = [w for w in workloads if w.labels and len(w.labels) >= self.group_size]
            logger.info(f"移除 {raw - len(workloads)} 条标签少于 {self.group_size} 的样本, 其无法用于 SFT")

        if max_queries:
            workloads = [w for w in workloads if len(w.queries) <= max_queries]
            logger.info(f"过滤查询多于 {max_queries} 的样本后剩余 {len(workloads)} 条样本")

        self.data = expand_workloads(workloads, extend=self.extend, generate=self.generate, group_size=self.group_size)

        if len(self.data) != len(workloads):
            logger.info(f"扩展数据集到 {len(self.data)} 条样本以用于多轮对话训练")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        w = self.data[index]
        # if self.db_config is None and self.db_config_d is not None:
        #     self.db_config = DBOption.from_dict(self.db_config_d)
        #     self.connections = self.db_config.connections

        # conn = self.connections.get(w.db)

        # 使用MMDatasetConverter处理数据
        converted_item = self.converter(
            to_data_sample(w, self.mm_config.enable, self.mm_config.enable_sql, self.prompt_type)
        )

        # 构建输入文本
        instruction = converted_item.get("instruction", "")
        input_text = converted_item.get("input", "")
        # output_text = converted_item.get("output", "")

        # 构建对话格式的消息
        messages = [{"role": "system", "content": "You are a helpful assistant."}]

        # 1. 构造第一个 user 消息
        if input_text:
            user_content = f"{instruction}\n{input_text}"
        else:
            user_content = instruction
        messages.append({"role": "user", "content": user_content})

        if self.multi_round and w.labels:
            # 2. 根据 labels 构造中间多轮对话消息.
            # 首先, 无法被 self.group_size 整除的尾部 labels 被忽略.
            # 其次, 当 rl 或 eval 时, 排除最后一组; 当用于 sft 时, 需包含全部 labels, 最后一组在下方 if self.generate 的 else 分支中插入.
            # sft: <|user|>user_content<|assistant|>label_1<|user|>continue_instruction<|assistant|>label_2...<|user|>continue_instruction<|assistant|>label_n
            # rl: <|user|>user_content<|assistant|>label_1<|user|>continue_instruction<|assistant|>label_2...<|user|>continue_instruction
            indexes = w.labels
            group_count = len(indexes) // self.group_size - 1
            for i in range(group_count):
                start = i * self.group_size
                end = start + self.group_size
                group = indexes[start:end]
                # assert len(group) == self.group_size
                group = [idx for idx in group if idx != IDX_GUARD]
                messages.append({"role": "assistant", "content": "\n".join(str(idx) for idx in group)})
                messages.append({"role": "user", "content": continue_prompt(group_size=self.group_size)})

        # 除最后一段模型输出外的整个 prompt
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_text = cast(str, prompt_text)
        if not self._logged:
            logger.info(f"prompt_text (无模型响应) 样本: {prompt_text}")
        # 分词
        prompt_encoding = self.tokenizer.__call__(prompt_text, return_tensors="pt")
        prompt_input_ids = cast("Tensor", prompt_encoding["input_ids"])

        if self.generate:
            # 用于 generate, 不包含回复
            result = {
                "input_ids": prompt_input_ids.squeeze(0),
                "attention_mask": cast("Tensor", prompt_encoding["attention_mask"]).squeeze(0),
            }
        else:  # sft, 需包含回复
            # __init__ 中的检查保证了此时 labels 至少有 group_size 个元素
            messages.append(
                {"role": "assistant", "content": "\n".join(str(idx) for idx in (w.labels[-self.group_size :]))}
            )

            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            full_text = cast(str, full_text)
            if not self._logged:
                logger.info(f"full_text (包含模型响应) 样本: {full_text.removeprefix(prompt_text)=}")
            encoding = self.tokenizer.__call__(full_text, return_tensors="pt")
            input_ids = cast("Tensor", encoding["input_ids"])
            # 标记输出 labels
            assert input_ids[0, : prompt_input_ids.shape[1]].equal(prompt_input_ids[0]), (
                "分词不匹配，请检查数据或tokenizer"
            )
            attention_mask = cast("Tensor", encoding["attention_mask"])
            labels: "Tensor" = input_ids.clone()
            labels[:, : prompt_input_ids.shape[1]] = -100  # ignore prompt部分

            result = {
                "input_ids": input_ids.squeeze(0),
                "attention_mask": attention_mask.squeeze(0),
                "labels": labels.squeeze(0),
            }

        # 添加列嵌入和SQL嵌入
        if DATASET_COLUMNS in converted_item:
            result[DATASET_COLUMNS] = converted_item[DATASET_COLUMNS]

        if DATASET_SQLS in converted_item:
            result[DATASET_SQLS] = converted_item[DATASET_SQLS]

        self._logged = True
        return result


def new_collator(pad_token_id: int):
    """带 `labels` 的 `collate_fn`, 用于 `sft`, 总是使用 `right padding`"""

    def fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        # padding: 分别使用 pad_token_id, 0, -100 对 input_ids, attention_mask, labels 进行 right side padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        result = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        # padding 列嵌入
        if DATASET_COLUMNS in batch[0]:
            columns = [item[DATASET_COLUMNS] for item in batch]
            result[DATASET_COLUMNS] = pad_sequence(columns, batch_first=True, padding_value=0)

        # 处理SQL嵌入
        if DATASET_SQLS in batch[0]:
            sqls = [item[DATASET_SQLS] for item in batch]
            result[DATASET_SQLS] = pad_sequence(sqls, batch_first=True, padding_value=0)

        return result

    return fn


def new_collator_rl(pad_token_id: int, padding_side: str = "left"):
    """不带 `labels` 的 `collate_fn`, 在生成阶段使用 `left padding`, 在训练阶段使用 `right padding`"""

    def fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        # 生成阶段: 分别使用 pad_token_id, 0 对 input_ids, attention_mask 进行 left padding
        # 训练阶段 right padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id, padding_side=padding_side)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0, padding_side=padding_side)

        result = {"input_ids": input_ids, "attention_mask": attention_mask}

        # padding 列嵌入
        if DATASET_COLUMNS in batch[0]:
            columns = [item[DATASET_COLUMNS] for item in batch]
            result[DATASET_COLUMNS] = pad_sequence(columns, batch_first=True, padding_value=0)

        # 处理SQL嵌入
        if DATASET_SQLS in batch[0]:
            sqls = [item[DATASET_SQLS] for item in batch]
            result[DATASET_SQLS] = pad_sequence(sqls, batch_first=True, padding_value=0)

        return result

    return fn


class SFTTrainer:
    """SFT投影层训练器"""

    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        finetuning_args: FinetuningArguments,
        generating_args: GeneratingArguments,
        workload_file: str,
        mm_config: MMConfig,
        db_option: DBOption | None,
        output_dir: str,
        max_length: int = 2048,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        epochs: int = 3,
        warmup_steps: int = 100,
        scheduler_type: Literal["const", "linear"] = "linear",
        save_steps: int = 100,
        log_steps: int = 10,
        gradient_accumulation_steps: int = 1,
        resume_from: Optional[str] = None,
        eval_data_path: Optional[str] = None,
        eval_set_percent: float = 0,
        train_extend: bool = True,
        eval_extend: bool = True,
        group_size: int = 1,
        prompt_type: Literal["general", "prompt1", "plus"] = "general",
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args

        self.output_dir = output_dir
        self.workload_file = workload_file
        self.eval_data_path = eval_data_path
        self.eval_set_percent = eval_set_percent
        self.mm_config = mm_config
        self.db_option = db_option
        self.max_length = max_length  # not used
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.scheduler_type = scheduler_type
        self.save_steps = save_steps
        self.log_steps = log_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.resume_from = resume_from
        self.train_extend = train_extend
        self.eval_extend = eval_extend
        self.group_size = group_size
        self.prompt_type: Literal["general", "prompt1", "plus"] = prompt_type

        # 分布式训练配置
        self.local_rank, self.global_rank, self.world_size = self._setup_distributed()
        self.device = torch.device(f"cuda:{self.local_rank}" if self.world_size > 1 else "cuda")

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.start_time = time.time()

        # 日志文件路径
        self.log_file_path = os.path.join(self.output_dir, "trainer_log.jsonl")

    def _setup_distributed(self) -> tuple[int, int, int]:
        """初始化分布式训练"""
        if "RANK" in os.environ:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            local_rank = int(os.environ["LOCAL_RANK"])
            return local_rank, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        else:
            return 0, 0, 1

    def _cleanup_distributed(self) -> None:
        """清理分布式训练"""
        if dist.is_initialized():
            dist.destroy_process_group()

    def _get_trainable_parameters(self, model: torch.nn.Module) -> list:
        """获取可训练的参数"""
        trainable_params = []
        projector_params = []
        lm_params = []

        # 获取实际的模型（如果是DDP包装的）
        actual_model = model.module if isinstance(model, DDP) else model

        for name, param in actual_model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                if "projector" in name.lower():
                    projector_params.append((name, param.numel()))
                else:
                    lm_params.append((name, param.numel()))

        if self.global_rank == 0:
            total_params = sum(p.numel() for p in trainable_params)
            projector_total = sum(p[1] for p in projector_params)
            lm_total = sum(p[1] for p in lm_params)

            logger.info(f"可训练参数总数: {total_params:,}")
            logger.info(f"  - 投影器参数: {projector_total:,} ({projector_total / total_params * 100:.1f}%)")
            logger.info(f"  - 语言模型参数: {lm_total:,} ({lm_total / total_params * 100:.1f}%)")

            # 打印投影器参数名称
            if projector_params:
                logger.info(f"投影器参数: {[name for name, _ in projector_params]}")

        return trainable_params

    def _write_log_to_file(self, log_data: dict[str, Any]) -> None:
        """将日志写入JSONL文件"""
        if self.global_rank != 0:
            return

        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

    def _format_time(self, seconds: float) -> str:
        """格式化时间为 H:MM:SS 格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}"

    def evaluate(self, model: torch.nn.Module, eval_dataloader: DataLoader) -> float:
        """在测试集上进行评估"""
        model.eval()
        total_eval_loss = 0.0
        num_eval_batches = len(eval_dataloader)

        with torch.no_grad():
            if self.global_rank == 0:
                eval_pbar = tqdm(eval_dataloader, desc="Evaluating")
            else:
                eval_pbar = eval_dataloader
            for batch in eval_pbar:
                # 将数据移到设备
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)

                # 前向传播
                outputs = model(**batch, return_dict=True)
                if isinstance(outputs, tuple):
                    loss = outputs[0]
                else:
                    loss = outputs.loss

                total_eval_loss += loss.item()
                if isinstance(eval_pbar, tqdm):
                    eval_pbar.set_postfix({"eval_loss": f"{loss.item():.4f}"})

        # 如果是分布式训练，需要同步所有进程的评估损失
        if self.world_size > 1:
            total_eval_loss_tensor = torch.tensor(total_eval_loss, device=self.device)
            dist.all_reduce(total_eval_loss_tensor, op=dist.ReduceOp.SUM)
            total_eval_loss = total_eval_loss_tensor.item() / self.world_size

        avg_eval_loss = total_eval_loss / num_eval_batches
        model.train()  # 切回训练模式
        return avg_eval_loss

    def _verify_parameter_updates(
        self,
        model: torch.nn.Module,
        lm_params_before: dict[str, torch.Tensor],
        projector_params_before: dict[str, torch.Tensor],
    ) -> None:
        """验证参数更新情况"""
        lm_params_changed = []
        lm_params_unchanged = []
        projector_params_changed = []
        projector_params_unchanged = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if "projector" in name.lower():
                if name in projector_params_before:
                    if torch.equal(param.data, projector_params_before[name]):
                        projector_params_unchanged.append(name)
                    else:
                        projector_params_changed.append(name)
            else:
                if name in lm_params_before:
                    if torch.equal(param.data, lm_params_before[name]):
                        lm_params_unchanged.append(name)
                    else:
                        lm_params_changed.append(name)

        # 报告验证结果
        if lm_params_changed:
            logger.info(f"✅ 变化的 lm params ({len(lm_params_changed)}): {lm_params_changed[:5]}...")
        if lm_params_unchanged:
            logger.info(f"❌ 不变的 lm params ({len(lm_params_unchanged)}): {lm_params_unchanged[:5]}...")

        if projector_params_changed:
            logger.info(
                f"✅ 变化的 projector params ({len(projector_params_changed)}): {projector_params_changed[:5]}..."
            )
        if projector_params_unchanged:
            logger.info(
                f"❌ 不变的 projector params ({len(projector_params_unchanged)}): {projector_params_unchanged[:5]}..."
            )

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        tokenizer: Any,
        loss: float,
        epoch: int,
        step: int,
    ) -> None:
        """保存检查点"""
        if self.global_rank != 0:
            return

        output_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(output_dir, exist_ok=True)

        # 如果是DDP模型，获取原始模型
        if isinstance(model, DDP):
            model_to_save = model.module
        else:
            model_to_save = model
        model_to_save = cast(PreTrainedModel, model_to_save)

        projector_state_dict = {}
        lm_state_dict = {}
        for name, param in model_to_save.named_parameters():
            if not param.requires_grad:
                continue
            if "projector" in name.lower():
                projector_state_dict[name] = param.cpu().clone()
            else:
                lm_state_dict[name] = param.cpu().clone()

        # 构建检查点数据
        checkpoint = {
            "epoch": epoch,
            "global_step": step,
            "projector_state_dict": projector_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
        }
        if lm_state_dict:
            checkpoint["lm_state_dict"] = lm_state_dict

        # 保存权重文件（safetensors格式）
        safetensors_path = os.path.join(output_dir, "projector.safetensors")
        save_file(projector_state_dict, safetensors_path)
        if lm_state_dict:
            # perf 会自动保存 adapter_model.safetensors 和 adapter_config.json
            model_to_save.save_pretrained(output_dir)

        # 保存完整检查点（PyTorch格式，包含优化器状态等）
        checkpoint_path = os.path.join(output_dir, "state.pt")
        torch.save(checkpoint, checkpoint_path)

        # 保存tokenizer（只保存一次）
        tokenizer_path = os.path.join(self.output_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            tokenizer.save_pretrained(self.output_dir)

        logger.info(f"检查点保存到: {checkpoint_path}")

        # 移除旧检查点中的 state.pt 文件以节省空间. 只需要移除倒数第二个检查点的 state.pt 即可.
        if step > self.save_steps:
            old_step = step - self.save_steps
            old_checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{old_step}")
            old_checkpoint_path = os.path.join(old_checkpoint_dir, "state.pt")
            if os.path.exists(old_checkpoint_path):
                try:
                    os.remove(old_checkpoint_path)
                    logger.info(f"已删除旧检查点文件: {old_checkpoint_path}")
                except Exception as e:
                    logger.warning(f"删除旧检查点文件失败: {old_checkpoint_path}, 错误: {e}")

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> tuple[int, int]:
        """加载检查点"""
        if self.global_rank == 0:
            logger.info(f"从检查点恢复训练: {checkpoint_path}")

        checkpoint = torch.load(os.path.join(checkpoint_path, "state.pt"), map_location=self.device)

        # 加载模型状态
        if isinstance(model, DDP):
            model_to_load = model.module
        else:
            model_to_load = model

        projector_state_dict = checkpoint["projector_state_dict"]
        lm_state_dict = checkpoint.get("lm_state_dict", {})
        current_model_dict = model_to_load.state_dict()

        # 过滤出存在于当前模型中的参数
        filtered_projector_state_dict = {k: v for k, v in projector_state_dict.items() if k in current_model_dict}
        filtered_lm_state_dict = {k: v for k, v in lm_state_dict.items() if k in current_model_dict}

        if len(filtered_projector_state_dict) != len(projector_state_dict):
            logger.warning(
                f"checkpoint projector_state_dict 中有 {len(projector_state_dict) - len(filtered_projector_state_dict)} 个参数在当前模型中不存在"
            )
        if lm_state_dict and len(filtered_lm_state_dict) != len(lm_state_dict):
            logger.warning(
                f"lm_state_dict 中有 {len(lm_state_dict) - len(filtered_lm_state_dict)} 个参数在当前模型中不存在"
            )

        current_model_dict.update(filtered_projector_state_dict)
        current_model_dict.update(filtered_lm_state_dict)
        model_to_load.load_state_dict(current_model_dict, strict=False)

        # 加载优化器和调度器状态
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as e:
            logger.warning(f"加载优化器/调度器状态失败: {e}")

        # 恢复训练状态
        epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        loss = checkpoint["loss"]

        if self.global_rank == 0:
            logger.info(f"恢复到 epoch {epoch}, step {global_step}, loss {loss:.4f}")

        return epoch, global_step

    def find_latest_checkpoint(self) -> Optional[str]:
        """查找最新的检查点"""
        if not os.path.exists(self.output_dir):
            return None

        checkpoint_dirs = []

        for p in os.listdir(self.output_dir):
            if p.startswith("checkpoint-"):
                try:
                    step_num = int(p.split("-")[1])
                    checkpoint_dirs.append((step_num, os.path.join(self.output_dir, p)))
                except (ValueError, IndexError):
                    continue

        if not checkpoint_dirs:
            return None

        # 返回步数最大的检查点
        latest_step, latest_dir = max(checkpoint_dirs, key=lambda x: x[0])

        if self.global_rank == 0:
            logger.info(f"找到最新检查点: {latest_dir} (step {latest_step})")

        return latest_dir

    def train_epoch(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        tokenizer: Any,
        total_steps: int,
        start_step: int = 0,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> float:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        # 获取实际的模型（如果是DDP包装的）
        actual_model = model.module if isinstance(model, DDP) else model

        if self.global_rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        else:
            pbar = dataloader
        for batch_idx, batch in enumerate(pbar):
            if batch_idx < start_step:
                continue
            self.global_step += 1

            # 在前向传播之前保存参数状态（仅在需要验证时）
            lm_params_before = {}
            projector_params_before = {}
            if self.global_rank == 0 and self.global_step <= 10:
                for name, param in actual_model.named_parameters():
                    if param.requires_grad:
                        if "projector" in name.lower():
                            projector_params_before[name] = param.data.clone()
                        else:
                            lm_params_before[name] = param.data.clone()

            # 将数据移到设备
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)

            # 前向传播
            outputs = model(**batch, return_dict=True)
            # 处理模型输出，可能是元组或带有loss属性的对象
            # logger.debug(f"{type(outputs)=}, {outputs=}")
            if isinstance(outputs, tuple):
                loss = outputs[0]  # 通常损失是元组的第一个元素
            else:
                loss = outputs.loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # 验证参数更新情况（仅在主进程的前几个步骤进行验证）
            if self.global_rank == 0 and self.global_step <= 10 and self.global_step > 1:
                self._verify_parameter_updates(actual_model, lm_params_before, projector_params_before)

            total_loss += loss.item()

            # 主进程显示进度条
            if isinstance(pbar, tqdm):
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{total_loss / (batch_idx - start_step + 1):.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        "step": self.global_step,
                    }
                )

            # 主进程记录日志
            if self.global_rank == 0 and self.log_steps > 0 and self.global_step % self.log_steps == 0:
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                avg_loss_val = total_loss / (batch_idx - start_step + 1)
                current_lr = scheduler.get_last_lr()[0]
                # 计算进度和预计剩余时间
                progress = self.global_step / total_steps
                remaining_time = elapsed_time / max(progress, 0.001) * (1 - progress) if progress > 0 else 0
                log_data = {
                    "type": "train",
                    "current_steps": self.global_step,
                    "total_steps": total_steps,
                    "loss": loss.item(),
                    "avg_loss": avg_loss_val,
                    "lr": current_lr,
                    "epoch": self.current_epoch + (batch_idx + 1) / num_batches,
                    "percentage": progress * 100,
                    "elapsed_time": self._format_time(elapsed_time),
                    "remaining_time": self._format_time(remaining_time),
                    "timestamp": datetime.now().isoformat(),
                }
                # 写入JSONL文件
                self._write_log_to_file(log_data)
                logger.info(
                    f"Step {self.global_step}: loss={loss.item():.4f}, "
                    f"avg_loss={avg_loss_val:.4f}, "
                    f"lr={current_lr:.2e}"
                )

            # 非主进程记录日志
            elif self.global_rank != 0 and self.log_steps > 0 and self.global_step % (self.log_steps * 5) == 0:
                logger.info(
                    f"Rank {self.local_rank} Step {self.global_step}: loss={loss.item():.4f}, "
                    f"avg_loss={total_loss / (batch_idx - start_step + 1):.4f}"
                )

            # 保存检查点
            if self.global_rank == 0 and self.save_steps > 0 and self.global_step % self.save_steps == 0:
                self.save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    tokenizer,
                    loss.item(),
                    self.current_epoch,
                    self.global_step,
                )

            # 进行评估
            if eval_dataloader is not None and self.save_steps > 0 and self.global_step % self.save_steps == 0:
                eval_loss = self.evaluate(model, eval_dataloader)
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                progress = self.global_step / total_steps
                remaining_time = elapsed_time / max(progress, 0.001) * (1 - progress) if progress > 0 else 0
                eval_log_data = {
                    "type": "eval",
                    "current_steps": self.global_step,
                    "total_steps": total_steps,
                    "eval_loss": eval_loss,
                    "epoch": self.current_epoch + (batch_idx + 1) / num_batches,
                    "percentage": progress * 100,
                    "elapsed_time": self._format_time(elapsed_time),
                    "remaining_time": self._format_time(remaining_time),
                    "timestamp": datetime.now().isoformat(),
                }
                self._write_log_to_file(eval_log_data)
                if self.global_rank == 0:
                    logger.info(f"Step {self.global_step} - Evaluation loss: {eval_loss:.4f}")

        return total_loss / max(1, num_batches - start_step)

    def train(self) -> None:
        """主训练函数"""
        if self.global_rank == 0:
            mm_config_dict = {k: v for k, v in self.mm_config.__dict__.items() if not isinstance(v, dict)}
            logger.debug(f"✅ mm_config: {mm_config_dict}")
        mm_init(self.mm_config)

        # 加载tokenizer和模型
        tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)
        tokenizer = hook_tokenizer(tokenizer)
        assert isinstance(tokenizer.pad_token_id, int)
        collate_fn = new_collator(tokenizer.pad_token_id)

        # 加载基础模型
        model = load_model(tokenizer, self.model_args, self.finetuning_args, self.training_args.do_train)
        # model = AutoModelForCausalLM.from_pretrained(
        #     self.model_path, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True
        # )
        # model = hook_load_model(model)
        # assert isinstance(model, MyModel)
        model = hook_sft_model(model)

        # 启用梯度检查点以减少显存占用
        if hasattr(model, "language_model") and hasattr(model.language_model, "gradient_checkpointing_enable"):
            model.language_model.gradient_checkpointing_enable()  # type: ignore
            if self.global_rank == 0:
                logger.info("已启用梯度检查点以减少显存占用")
        elif hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()  # type: ignore
            if self.global_rank == 0:
                logger.info("已启用梯度检查点以减少显存占用")

        # 将模型移到正确的设备
        model.to(self.device)  # type: ignore

        # 在创建DDP之前同步所有进程，确保模型状态一致
        if self.world_size > 1:
            dist.barrier()

        # 设置为分布式训练
        if self.world_size > 1:
            model = DDP(
                model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False
            )

        # 统计可训练参数
        model = cast(torch.nn.Module, model)
        self._get_trainable_parameters(model)

        # 创建数据集和数据加载器
        random.seed(0)
        if self.eval_set_percent > 0:
            workloads = load_workloads(self.workload_file)
            eval_size = int(len(workloads) * self.eval_set_percent)
            train_size = len(workloads) - eval_size
            if self.global_rank == 0:
                logger.info(f"拆分数据集，训练集大小: {train_size}, 评估集大小: {eval_size}")
            random.shuffle(workloads)
            train_set = workloads[:train_size]
            eval_set = workloads[train_size:]
            # 为 eval_set 创建摘要以跨训练验证其一致性. 确保当 dataset 和 eval_set_percent 相同时，eval_dataset 在不同运行中保持不变
            if self.global_rank == 0:
                eval_summary = compute_workloads_summary(eval_set)
                logger.info(f"评估数据集摘要: {eval_summary}")
        elif self.eval_data_path and os.path.exists(self.eval_data_path):
            train_set = load_workloads(self.workload_file)
            if self.global_rank == 0:
                logger.info(f"加载评估数据集: {self.eval_data_path}")
            eval_set = load_workloads(self.eval_data_path)
        else:
            train_set = load_workloads(self.workload_file)
            eval_set = None

        if self.prompt_type == "prompt1":
            assert self.db_option is not None
            if self.global_rank == 0:
                logger.info("启用 to_prompt1, 预计算列统计信息和谓词信息")
            init_schema_column_stats(self.db_option, workers=16)
            stats_targets = list(train_set)
            if eval_set:
                stats_targets += eval_set
            if self.global_rank == 0:
                logger.info(f"预计算谓词统计信息，数据集大小: {len(stats_targets)}")
            init_workload_predicate_stats(stats_targets, self.db_option, workers=16)

        # cache_workers = 16
        # logger.info(f"使用 {cache_workers} 个进程构建 cost/size 缓存")
        # t = time.time()
        # cost_cache, size_cache = build_cache(train_set, self.db_option, cache_workers)
        # logger.info(f"缓存构建完成，耗时 {time.time() - t:.2f} 秒")
        dataset = CustomDataset(
            train_set,
            tokenizer,
            self.mm_config,
            self.db_option.to_dict() if self.db_option else None,
            extend=self.train_extend,
            log_sample=self.global_rank == 0,
            # cost_cache=cost_cache,
            # size_cache=size_cache,
            group_size=self.group_size,
            prompt_type=self.prompt_type,
        )

        num_workers = 4
        persistent_workers = num_workers > 0

        # 训练配置
        if self.global_rank == 0:
            logger.info(
                f"最终训练配置: {self.mm_config.projector_trainable=}, {self.mm_config.projector2_trainable=}, {self.mm_config.lm_trainable=}"
            )

        sampler = DistributedSampler(dataset) if self.world_size > 1 else None
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
        )

        # 创建评估数据加载器
        eval_dataloader = None
        if eval_set:
            # logger.info(f"使用 {cache_workers} 个进程构建评估集 cost/size 缓存")
            # t = time.time()
            # cost_cache, size_cache = build_cache(eval_set, self.db_option, cache_workers)
            # logger.info(f"评估集缓存构建完成，耗时 {time.time() - t:.2f} 秒")
            eval_dataset = CustomDataset(
                eval_set,
                tokenizer,
                self.mm_config,
                self.db_option.to_dict() if self.db_option else None,
                extend=self.eval_extend,
                # cost_cache=cost_cache,
                # size_cache=size_cache,
                group_size=self.group_size,
                prompt_type=self.prompt_type,
            )
            eval_sampler = DistributedSampler(eval_dataset, shuffle=False) if self.world_size > 1 else None
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.batch_size,
                sampler=eval_sampler,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=persistent_workers,
            )

        # 设置优化器和调度器
        no_decay = ["bias", "layernorm", "rmsnorm"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n.lower() for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n.lower() for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        total_steps = len(dataloader) * self.epochs
        if self.scheduler_type == "const":
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=total_steps
            )
        assert scheduler is not None

        # 尝试从检查点恢复
        start_epoch = 0
        start_step = 0

        if self.resume_from:
            checkpoint_path = self.resume_from
            if os.path.exists(checkpoint_path):
                start_epoch, self.global_step = self.load_checkpoint(checkpoint_path, model, optimizer, scheduler)
                start_step = self.global_step % len(dataloader)
                start_epoch = self.global_step // len(dataloader)
        else:
            # 自动查找最新检查点
            latest_checkpoint = self.find_latest_checkpoint()
            if latest_checkpoint:
                start_epoch, self.global_step = self.load_checkpoint(latest_checkpoint, model, optimizer, scheduler)
                start_step = self.global_step % len(dataloader)
                start_epoch = self.global_step // len(dataloader)

        if self.global_rank == 0:
            logger.info(f"开始训练，共 {self.epochs} 个 epoch，每个 epoch {len(dataloader)} 个 batch")
            logger.info(f"总训练步数: {total_steps}, 预热步数: {self.warmup_steps}")
            if start_epoch > 0 or start_step > 0:
                logger.info(f"从 epoch {start_epoch}, step {start_step} 恢复训练")

        # 训练循环
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch: int = epoch + 1

            if sampler is not None:
                sampler.set_epoch(epoch)

            epoch_start_step = start_step if epoch == start_epoch else 0
            avg_loss = self.train_epoch(
                model, dataloader, optimizer, scheduler, tokenizer, total_steps, epoch_start_step, eval_dataloader
            )

            if self.global_rank == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs} 完成，平均损失: {avg_loss:.4f}")

                # 每个epoch结束时保存模型
                # self.save_checkpoint(model, optimizer, scheduler, tokenizer, avg_loss, epoch + 1, self.global_step)

        if self.global_rank == 0:
            logger.info("训练完成！")

            # 最终参数验证报告
            self._final_parameter_report(model)

        # 清理分布式训练
        self._cleanup_distributed()

    def _final_parameter_report(self, model: torch.nn.Module) -> None:
        """生成最终的参数验证报告"""
        actual_model = model.module if isinstance(model, DDP) else model

        trainable_projector_params = []
        trainable_lm_params = []
        frozen_projector_params = []
        frozen_lm_params = []

        for name, param in actual_model.named_parameters():
            if "projector" in name.lower():
                if param.requires_grad:
                    trainable_projector_params.append((name, param.numel()))
                else:
                    frozen_projector_params.append((name, param.numel()))
            else:
                if param.requires_grad:
                    trainable_lm_params.append((name, param.numel()))
                else:
                    frozen_lm_params.append((name, param.numel()))

        logger.info("=" * 60)
        logger.info("最终参数验证报告")
        logger.info("=" * 60)

        if trainable_projector_params:
            total_projector = sum(p[1] for p in trainable_projector_params)
            logger.info(f"✅ 可训练投影器参数: {len(trainable_projector_params)} 个, {total_projector:,} 参数")
            for name, count in trainable_projector_params:
                logger.info(f"  - {name}: {count:,}")

        if trainable_lm_params:
            total_lm = sum(p[1] for p in trainable_lm_params)
            logger.info(f"✅  可训练语言模型参数: {len(trainable_lm_params)} 个, {total_lm:,} 参数")
            for name, count in trainable_lm_params[:5]:  # 只显示前5个
                logger.info(f"  - {name}: {count:,}")
        else:
            logger.info("✅ 语言模型参数冻结")

        if frozen_projector_params:
            total_frozen_proj = sum(p[1] for p in frozen_projector_params)
            logger.info(f"🔒 冻结的投影器参数: {len(frozen_projector_params)} 个, {total_frozen_proj:,} 参数")

        logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 数据参数
    parser.add_argument("workload_file", help="SFT workload 文件路径 (需包含 labels 字段)", type=str)
    parser.add_argument("--eval-data-file", help="评估 workload 文件路径（可选）", type=str, default=None)
    parser.add_argument(
        "--eval-set-percent",
        help="评估集比例 (0-1), 将从 sft 数据集中取出一定比例的数据供评估使用",
        type=float,
        default=0.0,
    )
    parser.add_argument("--output-dir", help="输出目录", type=str, default="")
    # 训练参数
    parser.add_argument(
        "--adapter-dir", help="lora adapter path, 将覆盖配置文件中的 adapter_name_or_path", type=str, default=None
    )
    parser.add_argument("--epochs", help="训练轮数", type=int, default=3)
    parser.add_argument("--batch-size", help="批次大小", type=int, default=4)
    parser.add_argument("--learning-rate", help="学习率", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", help="预热步数", type=int, default=100)
    parser.add_argument(
        "--scheduler-type", "--sche", help="学习率调度器类型", choices=["linear", "const"], default="linear"
    )
    parser.add_argument("--save-steps", help="保存间隔, 同时作为评估间隔", type=int, default=100)
    parser.add_argument("--log-steps", help="日志打印间隔", type=int, default=10)
    parser.add_argument("--gradient-accumulation-steps", help="梯度累积步数", type=int, default=1)
    parser.add_argument("--resume-from", help="恢复训练的检查点路径", type=str, default=None)
    parser.add_argument("--train-extend", dest="train_extend", action="store_true", help="启用训练数据集的多轮扩展")
    parser.add_argument(
        "--no-train-extend", dest="train_extend", action="store_false", help="禁用训练数据集的多轮扩展"
    )
    parser.add_argument("--eval-extend", dest="eval_extend", action="store_true", help="启用评估数据集的多轮扩展")
    parser.add_argument("--no-eval-extend", dest="eval_extend", action="store_false", help="禁用评估数据集的多轮扩展")
    parser.set_defaults(train_extend=True, eval_extend=True)
    parser.add_argument("--group-size", "--gs", help="每轮推荐的索引数量", type=int, default=1)
    parser.add_argument("--ptype", choices=["general", "prompt1", "plus"], default="general", help="指定 prompt 变体")

    # 多模态参数
    MMConfig.add_arg_group_to(parser)
    # 数据库参数
    DBOption.add_arg_group_to(parser)

    parser.add_argument("--debug", help="调试模式", action="store_true")

    raw_args = sys.argv
    args = parse_args_safe(parser)

    # 参数检查
    if args.eval_set_percent < 0 or args.eval_set_percent >= 1:
        raise ValueError("--eval-set-percent 必须在 [0, 1) 范围内")
    if args.eval_set_percent > 0 and (args.eval_data_file is not None):
        raise ValueError("不能同时指定 --eval-data-file 和 --eval-set-percent")
    if args.group_size <= 0:
        raise ValueError("group_size 必须大于 0")

    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()
    if args.adapter_dir:
        if finetuning_args.finetuning_type != "lora":
            raise ValueError("--adapter-dir 只能用于 LoRA 微调")
        args.adapter_dir = resolve_ckpt(args.adapter_dir)
        model_args.adapter_name_or_path = [args.adapter_dir]  # type: ignore
        if not os.path.exists(args.adapter_dir):
            raise ValueError(f"指定的 adapter_dir 不存在: {args.adapter_dir}")
    elif model_args.adapter_name_or_path:
        raise ValueError("请通过 --adapter-dir 指定 LoRA adapter 路径, 配置文件中的 adapter_name_or_path 将被忽略")

    # 初始化多模态配置
    mm_config = MMConfig.from_args(args)
    if mm_config.lm_trainable:
        if finetuning_args.finetuning_type == "freeze":
            raise ValueError("设定为训练语言模型, 但配置文件指定 finetuning_type 为 freeze, 可能使用了错误的配置文件")

    if args.ptype == "prompt1":
        assert args.db is not None, "prompt1 模式需要指定 --db 参数以连接数据库"

    db_option = DBOption.from_args(args) if args.db is not None else None

    # 日志
    if args.output_dir:
        output_dir = args.output_dir
    else:
        assert training_args.output_dir, "output_dir 不能为空"
        output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if "RANK" not in os.environ or os.environ["RANK"] == "0":
        log_to_file("sft_projector", "DEBUG" if args.debug else "INFO", args=raw_args, log_dir=output_dir)

    # 创建训练器
    trainer = SFTTrainer(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        workload_file=args.workload_file,
        mm_config=mm_config,
        db_option=db_option,
        output_dir=output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        scheduler_type=args.scheduler_type,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        resume_from=args.resume_from,
        eval_data_path=args.eval_data_file,
        eval_set_percent=args.eval_set_percent,
        train_extend=args.train_extend,
        eval_extend=args.eval_extend,
        group_size=args.group_size,
        prompt_type=args.ptype,
    )

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()

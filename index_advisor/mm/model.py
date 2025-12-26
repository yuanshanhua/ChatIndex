import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_utils import get_last_checkpoint

from ..ia_logging import logger


logger = logger.getChild("mm_model")

if TYPE_CHECKING:
    from transformers.modeling_outputs import CausalLMOutputWithPast

COLUMN_TOKEN = "<|column|>"  # 列嵌入的占位 token
SQL_TOKEN = "<|sql|>"
DATASET_COLUMNS = "columns"  # 数据集中列字段的键名, 不要修改, 有其它地方硬编码了
DATASET_SQLS = "sqls"  # 数据集中 sql 字段的键名


@dataclass
class MMConfig:
    enable: bool = False
    enable_sql: bool = False
    column_embeddings: dict[str, torch.Tensor] = field(default_factory=dict)
    column_token_id: int = -1
    sql_embeddings: dict[str, torch.Tensor] = field(default_factory=dict)
    sql_token_id: int = -1
    column_embedding_size: int = 0
    sql_embedding_size: int = 0
    lm_trainable: bool = True  # 是否训练语言模型
    projector_trainable: bool = True
    projector2_trainable: bool = True  # projector2 代表 SQL projector
    projector_checkpoint: str | Path | None = None
    projector2_checkpoint: str | Path | None = None
    projector_dtype: torch.dtype = torch.bfloat16  # 投影层的类型, sft: float32, rl: bf16
    projector2_dtype: torch.dtype = torch.bfloat16  # sql 投影层的类型, sft: float32, rl: bf16

    def __post_init__(self):
        if self.projector_checkpoint and not self.enable:
            raise ValueError("MMConfig: 提供了 projector_checkpoint, 但 enable=False (未传递 --mm)")
        if self.projector2_checkpoint and not self.enable_sql:
            raise ValueError("MMConfig: 提供了 projector2_checkpoint, 但 enable_sql=False (未传递 --mm2)")

    def projector_dtype_(self):
        return self.projector_dtype

    def projector2_dtype_(self):
        return self.projector2_dtype

    def get_sql_embedding(self, sql: str) -> torch.Tensor:
        query_hash = hashlib.md5(sql.encode("utf-8")).hexdigest()
        if query_hash not in self.sql_embeddings:
            raise ValueError(f"SQL 嵌入不存在: {sql} (hash: {query_hash})")
        return self.sql_embeddings[query_hash]

    @classmethod
    def add_arg_group_to(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """添加多模态相关的参数到 argparse"""
        mmargs = parser.add_argument_group("MultiModal", "多模态选项")
        mmargs.add_argument("--mm", "--cmm", action="store_true", help="是否启用 column 多模态")
        mmargs.add_argument("--mm2", "--smm", action="store_true", help="是否启用 SQL 多模态")
        mmargs.add_argument(
            "--mm-embedding-files",
            "--mm-efs",
            "--cmm-efs",
            type=str,
            nargs="+",
            help="指定列嵌入文件路径, 使用空格分隔多个",
        )
        mmargs.add_argument(
            "--mm2-embedding-files",
            "--mm2-efs",
            "--smm-efs",
            type=str,
            nargs="+",
            help="指定 SQL 嵌入文件路径, 使用空格分隔多个",
        )
        mmargs.add_argument("--mm-freeze-lm", "--mm-flm", "--flm", action="store_true", help="是否冻结语言模型")
        mmargs.add_argument(
            "--mm-projector-trainable",
            "--mm-pt",
            "--cmm-pt",
            action="store_true",
            help="是否训练 column 投影层",
        )
        mmargs.add_argument(
            "--mm2-projector-trainable",
            "--mm2-pt",
            "--smm-pt",
            action="store_true",
            help="是否训练 SQL 投影层",
        )
        mmargs.add_argument(
            "--mm-projector-checkpoint",
            "--mm-pckpt",
            "--cmm-pckpt",
            type=str,
            default=None,
            help="column 投影层的 checkpoint 路径",
        )
        mmargs.add_argument(
            "--mm2-projector-checkpoint",
            "--mm2-pckpt",
            "--smm-pckpt",
            type=str,
            default=None,
            help="SQL 投影层的 checkpoint 路径",
        )
        return parser

    @classmethod
    def from_args(cls, args: "argparse.Namespace") -> "MMConfig":
        """从 argparse.Namespace 创建 MMConfig"""

        # 解析 projector checkpoint 路径, 允许 /last 表示最后一个 checkpoint
        def last_checkpoint(p: str) -> str:
            path = Path(p)
            if path.name == "last":  # xxx/last -> xxx/checkpoint-{max}
                r = get_last_checkpoint(path.parent)
                if r is None:
                    raise ValueError(f"未找到最后一个 checkpoint: {args.mm_projector_checkpoint}")
                path = Path(r)
            if path.is_dir():
                return str(path / "projector.safetensors")
            return p

        if args.mm_projector_checkpoint:
            args.mm_projector_checkpoint = last_checkpoint(args.mm_projector_checkpoint)
            logger.info(f"使用 mm projector checkpoint: {args.mm_projector_checkpoint}")

        if args.mm2_projector_checkpoint:
            args.mm2_projector_checkpoint = last_checkpoint(args.mm2_projector_checkpoint)
            logger.info(f"使用 mm2 projector checkpoint: {args.mm2_projector_checkpoint}")

        if args.mm:
            column_embeddings, column_embedding_size = cls._column_args(args)
        else:
            column_embeddings = {}
            column_embedding_size = 0

        if args.mm2:
            sql_embeddings, sql_embedding_size = cls._sql_args(args)
        else:
            sql_embeddings = {}
            sql_embedding_size = 0

        return cls(
            enable=args.mm,
            enable_sql=args.mm2,
            column_embeddings=column_embeddings,
            sql_embeddings=sql_embeddings,
            column_embedding_size=column_embedding_size,
            sql_embedding_size=sql_embedding_size,
            projector_trainable=args.mm_projector_trainable,
            projector2_trainable=args.mm2_projector_trainable,
            lm_trainable=not args.mm_freeze_lm,
            projector_checkpoint=args.mm_projector_checkpoint,
            projector2_checkpoint=args.mm2_projector_checkpoint,
        )

    @classmethod
    def _column_args(cls, args):
        assert isinstance(args.mm_embedding_files, list) and len(args.mm_embedding_files) > 0, (
            "请指定至少一个列嵌入文件路径, 使用 --mm-embedding-files / --mm-efs"
        )

        embeddings = {}
        for file_path in args.mm_embedding_files:
            with safe_open(file_path, framework="pt", device="cpu") as f:  # type: ignore
                for key in f.keys():
                    embeddings[key] = f.get_tensor(key)
                metadata = f.metadata()
                if isinstance(metadata, dict):
                    print(f"Metadata: {metadata}")
        logger.info(f"成功加载 {len(embeddings)} 个列嵌入: {str(list(embeddings.keys())):.100s}")

        # 添加别名支持
        aliases = [("tpch", "tpch1g")]
        alias_embeddings = {}
        for a, b in aliases:
            for k in embeddings.keys():
                if k.startswith(a + "."):
                    alias_embeddings[b + k[len(a) :]] = embeddings[k]
                elif k.startswith(b + "."):
                    alias_embeddings[a + k[len(b) :]] = embeddings[k]
        if alias_embeddings:
            embeddings.update(alias_embeddings)
            logger.info(f"添加 {len(alias_embeddings)} 个列嵌入别名: {list(alias_embeddings.keys())}")

        try:
            column_embedding_size = next(iter(embeddings.values())).shape[0]
        except Exception as e:
            raise ValueError("无法获取列嵌入的维度, 请检查列嵌入文件是否正确") from e

        logger.info(f"列嵌入的维度: {column_embedding_size}")
        assert all(emb.shape[0] == column_embedding_size for emb in embeddings.values()), "所有列嵌入的维度必须相同"

        if args.mm_projector_checkpoint:
            r = load_file(args.mm_projector_checkpoint)
            for k, v in r.items():
                if "multi_modal_projector.linear_1.weight" in k:
                    assert v.shape[1] == column_embedding_size, (
                        f"列嵌入和 projector 维度不匹配: {v.shape[1]=} vs {column_embedding_size=}"
                    )

        return embeddings, column_embedding_size

    @classmethod
    def _sql_args(cls, args):
        assert isinstance(args.mm2_embedding_files, list) and len(args.mm2_embedding_files) > 0, (
            "请指定至少一个 SQL 嵌入文件路径, 使用 --mm2-embedding-files / --mm2-efs"
        )

        embeddings = {}
        for file_path in args.mm2_embedding_files:
            with safe_open(file_path, framework="pt", device="cpu") as f:  # type: ignore
                for key in f.keys():
                    embeddings[key] = f.get_tensor(key)
                metadata = f.metadata()
                if isinstance(metadata, dict):
                    print(f"Metadata: {metadata}")
        logger.info(f"成功加载 {len(embeddings)} 个 SQL 嵌入")

        try:
            sql_embedding_size = next(iter(embeddings.values())).shape[0]
        except Exception as e:
            raise ValueError("无法获取 SQL 嵌入的维度, 请检查 SQL 嵌入文件是否正确") from e

        logger.info(f"SQL 嵌入的维度: {sql_embedding_size}")
        assert all(emb.shape[0] == sql_embedding_size for emb in embeddings.values()), "所有 SQL 嵌入的维度必须相同"

        if args.mm2_projector_checkpoint:
            r = load_file(args.mm2_projector_checkpoint)
            for k, v in r.items():
                if "multi_modal_projector2.linear_1.weight" in k:
                    assert v.shape[1] == sql_embedding_size, (
                        f"SQL 嵌入和 projector2 维度不匹配: {v.shape[1]=} vs {sql_embedding_size=}"
                    )

        return embeddings, sql_embedding_size


class MultiModalProjector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        lm_hidden_size: int,
        projector_hidden_act: str = "gelu",
        multimodal_projector_bias: bool = True,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, lm_hidden_size, bias=multimodal_projector_bias)
        self.act = ACT2FN[projector_hidden_act]
        self.linear_2 = nn.Linear(lm_hidden_size, lm_hidden_size, bias=multimodal_projector_bias)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class MyModel(PreTrainedModel, GenerationMixin):
    def __init__(self, language_model: PreTrainedModel, config: MMConfig):
        super().__init__(language_model.config)
        self.config.model_type = "qwen_table"
        self.language_model = language_model
        self.column_embeddings = config.column_embeddings
        self.column_token_id = config.column_token_id
        self.sql_embeddings = config.sql_embeddings
        self.sql_token_id = config.sql_token_id
        self.projector_dtype = config.projector_dtype
        self.projector2_dtype = config.projector2_dtype

        # column 多模态
        self.multi_modal_projector = MultiModalProjector(
            config.column_embedding_size, self.language_model.config.hidden_size
        ).to(self.language_model.device, dtype=config.projector_dtype)

        if config.projector_checkpoint and config.enable:
            # 加载 projector 的 checkpoint
            state_dict = load_file(config.projector_checkpoint)
            state_dict = {k.removeprefix("base_model.model."): v for k, v in state_dict.items()}
            state_dict = {
                k.removeprefix("multi_modal_projector."): v
                for k, v in state_dict.items()
                if k.startswith("multi_modal_projector.")
            }
            logger.debug(f"加载 multi_modal_projector checkpoint: {state_dict=}")
            res = self.multi_modal_projector.load_state_dict(state_dict)
            logger.debug(f"加载 multi_modal_projector: {res.unexpected_keys=}")
            logger.info(f"成功加载 multi_modal_projector: {config.projector_checkpoint}")

        # 确保 projector 的数据类型正确
        self.multi_modal_projector = self.multi_modal_projector.to(dtype=config.projector_dtype)
        logger.debug(f"Projector dtype: {config.projector_dtype}")

        # SQL 多模态
        self.multi_modal_projector2 = MultiModalProjector(
            config.sql_embedding_size, self.language_model.config.hidden_size
        ).to(self.language_model.device, dtype=config.projector2_dtype)

        if config.projector2_checkpoint and config.enable_sql:
            # 加载 projector2 的 checkpoint
            state_dict = load_file(config.projector2_checkpoint)
            state_dict = {k.removeprefix("base_model.model."): v for k, v in state_dict.items()}
            state_dict = {
                k.removeprefix("multi_modal_projector2."): v
                for k, v in state_dict.items()
                if k.startswith("multi_modal_projector2.")
            }
            logger.debug(f"加载 multi_modal_projector2 checkpoint: {state_dict=}")
            res = self.multi_modal_projector2.load_state_dict(state_dict)
            logger.debug(f"加载 multi_modal_projector2: {res.unexpected_keys=}")
            logger.info(f"成功加载 multi_modal_projector2: {config.projector2_checkpoint}")

        self.multi_modal_projector2 = self.multi_modal_projector2.to(dtype=config.projector2_dtype)
        logger.debug(f"Projector2 dtype: {config.projector2_dtype}")

        self.add_module("lm_head", self.language_model.lm_head)  # type: ignore

        if hasattr(self.language_model, "hf_device_map"):
            self.hf_device_map = self.language_model.hf_device_map

        self.post_init()

    @property
    def padding_side(self):
        return self._padding_side

    @padding_side.setter
    def padding_side(self, padding_side: str):
        if padding_side not in ["left", "right"]:
            raise ValueError(f"{padding_side} is not `left` or `right`.")
        self._padding_side = padding_side

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        columns: torch.Tensor | None = None,
        sqls: torch.Tensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # def add_detect_backward(name, tensor) -> None:
        #     def bwh(x):
        #         print("🔑 计算反向传播梯度:", name)

        #     tensor.register_hook(bwh)

        # text embedding
        column_token_positions = None
        sql_token_positions = None
        if input_ids is not None:
            # if len(input_ids.shape) == 1:  # [x, y]
            #     input_ids = input_ids.unsqueeze(0)  # [1, x, y]
            # logger.debug(f"⚠️ input_ids shape: {input_ids.shape}")
            # add_detect_backward("input_ids", input_ids)
            column_token_positions = input_ids == self.column_token_id
            # add_detect_backward("column_token_positions", column_token_positions)
            sql_token_positions = input_ids == self.sql_token_id
            if inputs_embeds is None:
                # 替换 <column> token id
                # tmp_input_ids = input_ids.clone()
                # add_detect_backward("tmp_input_ids", tmp_input_ids)
                input_ids.masked_fill_(column_token_positions, 1)  # 只要替换为任意合法的 token id 即可
                input_ids.masked_fill_(sql_token_positions, 1)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                # add_detect_backward("inputs_embeds", inputs_embeds)

        # table embedding
        if columns is not None and inputs_embeds is not None and column_token_positions is not None:
            # 确保 columns 的数据类型与 projector 匹配
            columns = columns.to(dtype=self.projector_dtype)
            # logger.debug(f"⚠️ columns shape: {columns.shape}")
            # if len(columns.shape) == 1:  # [x, y]
            #     columns = columns.unsqueeze(0)  # [1, x, y]

            # 使用 multi_modal_projector 将列嵌入映射到 lm_hidden_size
            lm_column_embeddings = self.multi_modal_projector(columns)  # 参与反向传播
            # add_detect_backward("lm_column_embeddings", lm_column_embeddings)

            # 确保 lm_column_embeddings 与 inputs_embeds 的数据类型匹配
            # lm_column_embeddings = lm_column_embeddings.to(dtype=inputs_embeds.dtype)

            # 替换 <column> 位置的嵌入
            # 计算每个位置对应的列索引
            positions_cumsum = column_token_positions.long().cumsum(dim=1) - 1
            # add_detect_backward("positions_cumsum", positions_cumsum)
            # 获取所有 True 位置的索引
            batch_idx, seq_idx = column_token_positions.nonzero(as_tuple=True)
            # add_detect_backward("batch_idx", batch_idx)
            # add_detect_backward("seq_idx", seq_idx)
            col_idx = positions_cumsum[column_token_positions]
            # add_detect_backward("col_idx", col_idx)
            # 批量替换嵌入
            # logger.debug(f"⚠️ inputs_embeds shape: {inputs_embeds.shape}")
            # logger.debug(f"⚠️ lm_column_embeddings shape: {lm_column_embeddings.shape}")
            # logger.debug(f"⚠️ {batch_idx=} {seq_idx=} {col_idx=}")
            inputs_embeds[batch_idx, seq_idx] = lm_column_embeddings[batch_idx, col_idx]
            # 检查正确性
            # for i in range(inputs_embeds.shape[0]):
            #     k = 0
            #     for j in range(inputs_embeds.shape[1]):
            #         if positions[i, j]:
            #             assert inputs_embeds[i, j].equal(lm_column_embeddings[i, k])
            #             k += 1

        # sql embedding
        if sqls is not None and inputs_embeds is not None and sql_token_positions is not None:
            # 确保 sqls 的数据类型与 projector2 匹配
            sqls = sqls.to(dtype=self.projector2_dtype)
            # logger.debug(f"⚠️ sqls shape: {sqls.shape}")
            # if len(sqls.shape) == 1:  # [x, y]
            #     sqls = sqls.unsqueeze(0)  # [1, x, y]

            # 使用 multi_modal_projector2 将 sql 嵌入映射到 lm_hidden_size
            lm_sql_embeddings = self.multi_modal_projector2(sqls)  # 参与反向传播
            # add_detect_backward("lm_sql_embeddings", lm_sql_embeddings)
            # 确保 lm_sql_embeddings 与 inputs_embeds 的数据类型匹配
            # lm_sql_embeddings = lm_sql_embeddings.to(dtype=inputs_embeds.dtype)
            # 替换 <sql> 位置的嵌入
            # 计算每个位置对应的 sql 索引
            positions_cumsum = sql_token_positions.long().cumsum(dim=1) - 1
            # add_detect_backward("positions_cumsum", positions_cumsum)
            # 获取所有 True 位置的索引
            batch_idx, seq_idx = sql_token_positions.nonzero(as_tuple=True)
            # add_detect_backward("batch_idx", batch_idx)
            # add_detect_backward("seq_idx", seq_idx)
            sql_idx = positions_cumsum[sql_token_positions]
            # add_detect_backward("sql_idx", sql_idx)
            # 批量替换嵌入
            # logger.debug(f"⚠️ inputs_embeds shape: {inputs_embeds.shape}")
            # logger.debug(f"⚠️ lm_sql_embeddings shape: {lm_sql_embeddings.shape}")
            # logger.debug(f"⚠️ {batch_idx=} {seq_idx=} {sql_idx=}")
            inputs_embeds[batch_idx, seq_idx] = lm_sql_embeddings[batch_idx, sql_idx]
            # 检查正确性
            # for i in range(inputs_embeds.shape[0]):
            #     k = 0
            #     for j in range(inputs_embeds.shape[1]):
            #         if positions[i, j]:
            #             assert inputs_embeds[i, j].equal(lm_sql_embeddings[i, k])
            #             k += 1

        outputs: "CausalLMOutputWithPast" = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # 在调试时使用以下代码检查上述两个分支的差异. 当数据预处理正确使用 -100 padding label 时, 两个分支的 loss 应该完全相同
            # import torch.nn.functional as F; std_loss, filt_loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, logits.shape[-1]), labels[..., 1:].reshape(-1), ignore_index=-100), F.cross_entropy(logits[..., :-1, :][shift_attention_mask != 0], labels[..., 1:][shift_attention_mask != 0], ignore_index=-100); print(f"Loss: {std_loss:.4f} vs {filt_loss:.4f} (diff: {abs(std_loss-filt_loss):.6f})")
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.language_model.set_output_embeddings(value)  # type: ignore

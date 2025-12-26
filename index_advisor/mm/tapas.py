from typing import Iterable, Optional

import pandas as pd
import torch
from transformers import TapasModel, TapasTokenizer

from ..db import DBOption
from ..ia_logging import logger


logger = logger.getChild("tapas")


def from_schema(db: DBOption, sample_count: int) -> dict[str, pd.DataFrame]:
    """
    从数据库模式创建 pandas DataFrame 列表, 为每一列生成单独的 DataFrame 用于 TAPAS 模型输入

    Args:
        db: 数据库选项对象
        sample_count: 每个表的样本数量

    Returns:
        列标识符及对应的 pandas DataFrame 字典, 每个 DataFrame 代表一列数据
    """
    column_dataframes = {}

    for tname, table in db.schema.tables.items():
        # 获取表格样本数据
        samples = table.get_sample_value(db.conn, sample_count)
        assert len(samples) > 0, f"table {tname} has no sample value"
        assert len(samples[0]) == len(table.ordered_column_names), (
            f"{len(samples[0])=} but {len(table.ordered_column_names)=}"
        )
        logger.debug(f"get {len(samples)} samples for table {tname}")

        # 为每一列创建单独的 DataFrame
        for col_index, cname in enumerate(table.ordered_column_names):
            # 收集该列的所有值
            column_values = []
            for sample_row in samples:
                value = sample_row[col_index]
                # 处理空值, 转换为字符串
                if value is None:
                    value = ""
                else:
                    value = str(value)
                column_values.append(value)

            # 创建单列 DataFrame
            df_data = {cname: column_values}
            df = pd.DataFrame(df_data)

            # 设置列的元数据
            column_key = f"{db.db}.{tname}.{cname}"
            df.attrs["table_name"] = tname
            df.attrs["column_name"] = cname
            df.attrs["db_name"] = db.db

            # 添加列的详细元数据
            column = table.columns[cname]
            df.attrs["column_metadata"] = {
                "type": column.typ,
                "is_primary_key": (cname in table.primary_key_columns),
                "foreign_key": column.foreign_key.table + "." + column.foreign_key.name
                if column.foreign_key
                else None,
            }

            column_dataframes[column_key] = df
            logger.debug(f"created DataFrame for column {column_key} with shape {df.shape}")

    return column_dataframes


class TapasEmbeddingGenerator:
    """使用 TAPAS 模型生成表格嵌入的类."""

    def __init__(
        self,
        model_name: str = "google/tapas-base",
        device: Optional[str] = None,
        max_batch_rows: int = 200,
    ):
        """
        初始化 TAPAS 嵌入器

        Args:
            model_name: TAPAS 模型名称. large 模型有问题, 生成的列向量几乎一致, 不要使用.
            device: 计算设备 ('cpu', 'cuda', 'auto')
            max_batch_rows: 每次生成最多使用的行数, 防止模型输入过长. 超过该行数时会分批处理.
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_batch_rows = max_batch_rows

        # 加载模型和分词器
        try:
            self.tokenizer: TapasTokenizer = TapasTokenizer.from_pretrained(model_name)
            self.model = TapasModel.from_pretrained(model_name)
            self.model.to(self.device)  # type: ignore
            self.model.eval()
            logger.info(f"TAPAS 模型 {model_name} 加载成功, 使用设备: {self.device}")
        except Exception as e:
            logger.error(f"加载 TAPAS 模型失败: {e}")
            raise

    def _effective_sample_count(self, requested: int) -> int:
        """确定单表采样行数; 超过 max_batch_rows 时改为分批而非直接截断."""

        if requested <= 0:
            raise ValueError("sample_count must be positive")

        if requested > self.max_batch_rows:
            logger.warning("采样行数 %s 超过上限 %s, 将通过分批处理避免输入超长", requested, self.max_batch_rows)

        return requested

    def _preprocess_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """
        预处理表格数据

        Args:
            table: 输入的 pandas DataFrame

        Returns:
            预处理后的表格
        """
        # 处理空值
        table = table.fillna("")

        # 确保所有列都是字符串类型
        for col in table.columns:
            table[col] = table[col].astype(str)

        return table

    @staticmethod
    def _get_column_embeddings(
        inputs: dict[str, torch.Tensor], last_hidden_states: torch.Tensor
    ) -> list[torch.Tensor]:
        """Get the column embeddings from the last hidden states of a TAPAS model.

        Args:
            inputs: The inputs to the model.
            last_hidden_states: The last hidden states of the model.

        Returns:
            column_embeddings: The column embeddings.
        """
        column_ids = inputs["token_type_ids"][0][:, 1]
        # TAPAS 中列编号从 0 开始, 负数通常用于 [CLS]/问题部分, 这里过滤掉
        valid_column_ids = torch.unique(column_ids[column_ids >= 0]).tolist()

        column_embeddings = []

        for column_id in sorted(valid_column_ids):
            indices = torch.nonzero(column_ids == column_id, as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                continue

            embeddings = last_hidden_states[0][indices]
            column_embedding = embeddings.mean(dim=0)

            column_embeddings.append(column_embedding)

        return column_embeddings

    def _get_table_embedding(self, inputs: dict[str, torch.Tensor], last_hidden_states: torch.Tensor) -> torch.Tensor:
        """获取表级别的嵌入 (CLS token embedding)."""
        cls_positions = torch.nonzero(inputs["input_ids"][0] == self.tokenizer.cls_token_id, as_tuple=False)
        if cls_positions.numel() == 0:
            raise ValueError("CLS token not found in inputs")
        cls_index = cls_positions[0].item()
        return last_hidden_states[0, cls_index, :]

    def _generate_embeddings(
        self, table: pd.DataFrame, question: str = "", include_table_embedding: bool = False
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """生成列级和可选的表级嵌入."""

        try:
            processed_table = self._preprocess_table(table)

            total_rows = len(processed_table)
            if total_rows == 0:
                raise ValueError("空表无法生成嵌入")

            batch_size = self.max_batch_rows

            def _forward(batch_df: pd.DataFrame) -> tuple[list[torch.Tensor], torch.Tensor | None]:
                inputs = self.tokenizer(
                    table=batch_df,
                    queries=question if question else "",
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                inputs_device = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs_device)
                    last_hidden_states = outputs.last_hidden_state

                    column_embeddings = self._get_column_embeddings(inputs_device, last_hidden_states)
                    table_embedding = (
                        self._get_table_embedding(inputs_device, last_hidden_states)
                        if include_table_embedding
                        else None
                    )

                    return column_embeddings, table_embedding

            if total_rows <= batch_size:
                return _forward(processed_table)

            num_batches = (total_rows + batch_size - 1) // batch_size
            logger.info(f"表格行数 {total_rows} 超过上限 {batch_size}, 将分为 {num_batches} 个批次生成嵌入")

            col_sums: list[torch.Tensor] | None = None
            weight_sum = 0
            table_sum: torch.Tensor | None = None

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_rows)
                batch_df = processed_table.iloc[start_idx:end_idx].copy()
                batch_df.reset_index(drop=True, inplace=True)

                col_embeds, tbl_embed = _forward(batch_df)
                batch_rows = len(batch_df)
                if not col_embeds:
                    logger.warning(f"批次 {batch_idx + 1} 未生成列嵌入, 跳过")
                    continue

                if col_sums is None:
                    col_sums = [emb * batch_rows for emb in col_embeds]
                else:
                    if len(col_embeds) != len(col_sums):
                        raise ValueError("批次间的列数不一致, 无法合并嵌入")
                    for idx, emb in enumerate(col_embeds):
                        col_sums[idx] = col_sums[idx] + emb * batch_rows

                if include_table_embedding and tbl_embed is not None:
                    table_sum = tbl_embed * batch_rows if table_sum is None else table_sum + tbl_embed * batch_rows

                weight_sum += batch_rows

            if col_sums is None or weight_sum == 0:
                raise ValueError("未生成任何批次的列嵌入")

            col_avgs = [col_sum / weight_sum for col_sum in col_sums]
            table_embedding = table_sum / weight_sum if include_table_embedding and table_sum is not None else None

            return col_avgs, table_embedding

        except Exception as e:
            logger.error(f"生成表格嵌入时出错: {e}")
            raise

    def _generate_column_embeddings(self, table: pd.DataFrame, question: str = "") -> list[torch.Tensor]:
        """生成列级嵌入列表."""

        column_embeddings, _ = self._generate_embeddings(table, question=question, include_table_embedding=False)
        return column_embeddings

    def generate_table_embedding(self, table: pd.DataFrame, question: str = "") -> torch.Tensor:
        """生成表级别的嵌入向量 (CLS token)."""

        _, table_embedding = self._generate_embeddings(table, question=question, include_table_embedding=True)
        assert table_embedding is not None
        return table_embedding

    def generate_column_embeddings_from_db(self, db: DBOption, sample_count: int = 100) -> dict[str, torch.Tensor]:
        """
        从数据库生成列级嵌入。

        Args:
            db: 数据库选项对象
            sample_count: 每个表的样本数量

        Returns:
            列标识符到列嵌入的映射
        """
        effective_sample_count = self._effective_sample_count(sample_count)
        all_tables = from_schema(db, effective_sample_count)
        embeddings = {}

        for key, table in all_tables.items():
            logger.info(f"生成嵌入: {table.attrs}")
            embedding_list = self._generate_column_embeddings(table)
            if not embedding_list:
                logger.warning(f"表 {key} 未生成列嵌入, 跳过")
                continue
            embeddings[key] = embedding_list[0]

        return embeddings


if __name__ == "__main__":
    embedder = TapasEmbeddingGenerator("models/tapas-base")
    tables = {
        "col1": pd.DataFrame({"col1": ["a", "b", "c"]}),
        "col2": pd.DataFrame({"col2": ["x", "y", "z"]}),
    }

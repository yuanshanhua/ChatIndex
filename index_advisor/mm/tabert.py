import os
from typing import Dict, Literal

import torch

from TaBERT.table_bert import Column, Table

from ..db import DBOption
from ..ia_logging import logger


logger = logger.getChild("tabert")


def from_schema(db: DBOption, sample_count: int):
    tb_tables: list[Table] = []
    all_columns: dict[str, dict[str, Column]] = {}
    for tname, table in db.schema.tables.items():
        tb_columns: dict[str, Column] = {}
        # 构造 TaBERT Column
        for cname, column in table.columns.items():
            tb_column = Column(
                name=cname,
                type=column.tb_type,
                is_primary_key=(cname in table.primary_key_columns),
            )
            tb_columns[cname] = tb_column
        all_columns[tname] = tb_columns
    # 此时所有列都已创建相应的 TaBERT Column 对象
    for tname, table in db.schema.tables.items():
        # 设置外键
        for cname, column in table.columns.items():
            if column.foreign_key is None:
                continue
            logger.debug(f"[set foreign key] {tname}.{cname} -> {column.foreign_key.table}.{column.foreign_key.name}")
            all_columns[tname][cname].foreign_key = all_columns[column.foreign_key.table][column.foreign_key.name]
        # get table sample value
        samples = table.get_sample_value(db.conn, sample_count)
        assert len(samples) > 0, f"table {tname} has no sample value"
        assert len(samples[0]) == len(table.ordered_column_names), (
            f"{len(samples[0])=} but {len(table.ordered_column_names)=}"
        )
        logger.debug(f"get {len(samples)} samples for table {tname}")
        # set column sample value
        for cname, value in zip(table.ordered_column_names, samples[0]):
            all_columns[tname][cname].sample_value = value
            logger.debug(f"{tname}.{cname} sample value: {value}")
        # 构造 TaBERT Table
        tb_table = Table(
            name=tname,  # 这个字段似乎没用, 真正参与计算的是 id
            id=f"{tname}",
            header=[all_columns[tname][cname] for cname in table.ordered_column_names],
            data=[list(r) for r in samples],
        )
        assert [col.name for col in tb_table.header] == table.ordered_column_names, (
            f"TaBERT 列顺序与查询结果不一致: {[col.name for col in tb_table.header]} vs {table.ordered_column_names}"
        )
        tb_tables.append(tb_table)
    assert len(tb_tables) == len(db.schema.tables), f"表格数量不一致: {len(tb_tables)} vs {len(db.schema.tables)}"
    return tb_tables


class TaBertColumnEmbeddingGenerator:
    """使用 TaBERT 模型生成列嵌入"""

    def __init__(
        self,
        model_size: Literal["base", "large"] = "base",
        vertical: bool = False,
        model_path: str = "",
        base_model: str = "",
    ):
        """
        初始化 TaBERT 列嵌入生成器

        Args:
            model_size: TaBERT 模型大小，"base" 或 "large"
            vertical: 是否使用 vertical attention，True 对应 k3，False 对应 k1
            model_path: 模型路径，如果为空则使用默认路径
            base_model_name: 基础模型名称，如果为空则使用默认名称
        """
        # 设置默认路径
        if model_path == "":
            kn = "k3" if vertical else "k1"
            model_path = f"models/tabert/{model_size}_{kn}/model.bin"

        if base_model == "":
            base_model = f"models/bert-{model_size}-uncased"

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TaBERT 模型文件不存在: {model_path}, 请先下载或训练模型")

        # 加载模型
        from TaBERT.table_bert import TableBertModel

        self.model = TableBertModel.from_pretrained(model_path, base_model_name=base_model)
        self.tokenizer = self.model.tokenizer
        self.hidden_size = self.model.config.hidden_size

    def generate_column_embeddings(
        self, tables: list[Table], contexts=None, db_name: str = ""
    ) -> Dict[str, torch.Tensor]:
        contexts = [[""] for _ in tables]
        context_encoding, column_encoding, info_dict = self.model.encode(contexts, tables)
        # column_encoding: [num_tables, num_columns_in_table, *]
        expected = (len(tables), max(len(table.header) for table in tables), self.hidden_size)
        assert column_encoding.shape == expected, f"embedding shape: {column_encoding.shape}, {expected=}"

        res = {}
        for i, table in enumerate(tables):
            for j, column in enumerate(table.header):
                full_name = f"{db_name}.{table.name}.{column.name}"
                res[full_name] = column_encoding[i, j]
                print(f"column {full_name} embedding: {res[full_name].shape}")
        return res

    def from_db(self, db_option: DBOption, sample_count: int) -> Dict[str, torch.Tensor]:
        inputs = [table.tokenize(self.tokenizer) for table in from_schema(db_option, sample_count)]
        return self.generate_column_embeddings(inputs, db_name=db_option.databases[0])

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .ia_logging import logger
from .schemas import ColumnCoder, Index, schemas


if TYPE_CHECKING:
    from .sql_workload_features import WorkloadPredicateStats


logger = logger.getChild("sql")


@dataclass
class QueryInfo:
    """表示一个 SQL 查询的相关信息."""

    sql: str
    tables_columns: dict[str, set[str]]  # SQL 语句涉及的表和列
    large_columns: dict[str, set[str]]  # 不能用于索引的大字段及其所属表
    exist_indexes: set[Index]  # 相关列上已存在的索引
    basic_cost_explain: float  # 使用 explain 得到的基础 cost
    basic_cost_real: float | None = None  # 使用 explain analyze 得到的实际查询时间
    frequency: int = 1  # 查询在 workload 中的频率

    def __post_init__(self):
        self.basic_cost = self.basic_cost_real or self.basic_cost_explain

    def with_frequency(self, frequency: int) -> "QueryInfo":
        """返回一个新的 QueryInfo 对象，频率为 frequency."""
        return QueryInfo(
            sql=self.sql,
            tables_columns=self.tables_columns,
            large_columns=self.large_columns,
            exist_indexes=self.exist_indexes,
            basic_cost_explain=self.basic_cost_explain,
            basic_cost_real=self.basic_cost_real,
            frequency=frequency,
        )

    def __hash__(self) -> int:
        return hash((self.sql, self.frequency))

    def __lt__(self, other: "QueryInfo") -> bool:
        return self.sql < other.sql

    def to_dict(self):
        """将 QueryInfo 转换为可序列化的字典"""
        result = {
            "sql": self.sql,
            "tables_columns": {k: sorted(v) for k, v in sorted(self.tables_columns.items())},
            "large_columns": {k: sorted(v) for k, v in sorted(self.large_columns.items())},
            "exist_indexes": [{"table": idx.table, "columns": idx.columns} for idx in sorted(self.exist_indexes)],
            "basic_cost_explain": self.basic_cost_explain,
            "frequency": self.frequency,
        }
        if self.basic_cost_real is not None:
            result["basic_cost_real"] = self.basic_cost_real
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """从字典创建 QueryInfo 对象"""
        tables_columns = {k: set(v) for k, v in data.get("tables_columns", {}).items()}
        large_columns = {k: set(v) for k, v in data.get("large_columns", {}).items()}
        exist_indexes = {Index(idx["table"], idx["columns"]) for idx in data.get("exist_indexes", [])}
        basic_cost_real = data.get("basic_cost_real", None)

        return cls(
            sql=data["sql"],
            tables_columns=tables_columns,
            large_columns=large_columns,
            exist_indexes=exist_indexes,
            basic_cost_explain=data.get("basic_cost_explain", 0),
            basic_cost_real=basic_cost_real,
            frequency=data.get("frequency", 1),
        )


class Workload:
    def __init__(
        self,
        db: str,
        id: int,
        queries: set[QueryInfo],
        basic_cost_explain: float | None = None,
        basic_cost_real: float | None = None,
        # labels for sft.
        labels: Sequence[Index] | None = None,
        label_overrides: dict[int, Index] | None = None,
    ) -> None:
        self.db = db
        self.id = id
        self.queries = queries
        self.sqls = [q.sql for q in queries]
        self.sqls.sort()
        self.predicate_stats: "WorkloadPredicateStats | None" = None
        self.labels = list(labels) if labels is not None else None
        self.label_overrides: dict[int, Index] = {}
        if label_overrides:
            self.label_overrides.update({int(pos): idx for pos, idx in label_overrides.items()})
        if self.labels is None:
            self.label_overrides.clear()
        else:
            max_index = len(self.labels)
            self.label_overrides = {pos: idx for pos, idx in self.label_overrides.items() if 0 <= pos < max_index}
        self.tables_columns: dict[str, set[str]] = {}
        self.large_columns: dict[str, set[str]] = {}
        self.exist_indexes: set[Index] = set()
        self.basic_cost_explain = basic_cost_explain or sum(q.basic_cost_explain * q.frequency for q in queries)
        if basic_cost_real is not None:
            self.basic_cost_real: float | None = basic_cost_real
        elif all(q.basic_cost_real is not None for q in queries):
            self.basic_cost_real = sum((q.basic_cost_real or 0) * q.frequency for q in queries)
        else:
            self.basic_cost_real = None
        for q in queries:
            for table, columns in q.tables_columns.items():
                self.tables_columns.setdefault(table, set()).update(columns)
            for table, columns in q.large_columns.items():
                self.large_columns.setdefault(table, set()).update(columns)
            self.exist_indexes.update(q.exist_indexes)

    @cached_property
    def sqls_hash(self) -> str:
        """返回此 workload 中所有查询的 SQL 字符串的哈希值."""
        m = hashlib.md5()
        for sql in sorted(self.sqls):
            m.update(sql.encode("utf-8"))
        return m.hexdigest()

    @property
    def basic_cost(self) -> float:
        return self.basic_cost_real or self.basic_cost_explain

    @cached_property
    def column_ids(self) -> list[str]:
        """返回此 workload 中所有查询包含的所有列的全限定名称."""
        return sorted(
            f"{self.db}.{table}.{column}" for table, columns in self.tables_columns.items() for column in columns
        )

    @cached_property
    def column_ids_plus(self) -> list[str]:
        """返回此 workload 中所有查询包含的所有列的全限定名称, 且每个表开头插入多列嵌入名称 col1+col2+..."""
        res = []
        for table in sorted(self.tables_columns):
            all_columns = sorted(self.tables_columns[table])
            res.append("+".join(f"{self.db}.{table}.{column}" for column in all_columns))
            for column in all_columns:
                res.append(f"{self.db}.{table}.{column}")
        return res

    @cached_property
    def total_sql_length(self) -> int:
        """返回此 workload 中所有查询的SQL字符串总长度."""
        return sum(len(q.sql) for q in self.queries)

    @cached_property
    def total_table_count(self) -> int:
        """返回此 workload 中涉及的总表数."""
        return len(self.tables_columns)

    @cached_property
    def total_column_count(self) -> int:
        """返回此 workload 中涉及的总列数."""
        return sum(len(columns) for columns in self.tables_columns.values())

    @cached_property
    def total_query_count(self) -> int:
        """返回此 workload 中查询的总数量（考虑频率）."""
        return sum(q.frequency for q in self.queries)

    @cached_property
    def unique_query_count(self) -> int:
        """返回此 workload 中唯一查询的数量."""
        return len(self.queries)

    @cached_property
    def total_index_count(self) -> int:
        """返回此 workload 中已存在索引的数量."""
        return len(self.exist_indexes)

    @cached_property
    def avg_sql_length(self) -> float:
        """返回此 workload 中查询的平均SQL长度."""
        if not self.queries:
            return 0.0
        return sum(len(q.sql) for q in self.queries) / len(self.queries)

    @cached_property
    def sprint_tables_columns(self) -> str:
        """以多行字符串形式返回此 workload 中所有查询涉及的所有表和列, 用于 prompt 生成."""
        return "\n".join(
            sorted(f"{table}({','.join(sorted(columns))})" for table, columns in self.tables_columns.items())
        )

    @cached_property
    def sprint_large_columns(self) -> str:
        return "\n".join(
            sorted(f"{table}({','.join(sorted(columns))})" for table, columns in self.large_columns.items())
        )

    @cached_property
    def sprint_exist_indexes(self) -> str:
        return "\n".join(sorted(str(index) for index in self.exist_indexes))

    @cached_property
    def sprint_sqls(self) -> str:
        return "\n".join(self.sqls)

    @property
    def sprint_labels(self) -> str:
        if not self.labels:
            return ""
        return "\n".join(str(index) for index in self.labels)

    def label_at(self, position: int) -> Index | None:
        if self.labels is None or position < 0 or position >= len(self.labels):
            return None
        return self.label_overrides.get(position, self.labels[position])

    def effective_labels(self) -> list[Index]:
        if not self.labels:
            return []
        return [self.label_overrides.get(i, idx) for i, idx in enumerate(self.labels)]

    def with_labels(self, labels: Iterable[Index]) -> "Workload":
        """返回一个新的 Workload 对象，包含指定的标签."""
        w = Workload(
            db=self.db,
            id=self.id,
            queries=self.queries,
            basic_cost_explain=self.basic_cost_explain,
            basic_cost_real=self.basic_cost_real,
            labels=list(labels),
            label_overrides=self.label_overrides.copy(),
        )
        w.predicate_stats = self.predicate_stats
        return w

    def to_dict(self):
        """将 Workload 转换为可序列化的字典"""
        result: dict[str, Any] = {
            "db": self.db,
            "id": self.id,
            "sqls_hash": self.sqls_hash,
            "queries": [q.to_dict() for q in sorted(self.queries)],
            # "tables_columns": {k: sorted(v) for k, v in sorted(self.tables_columns.items())},
            # "large_columns": {k: sorted(v) for k, v in sorted(self.large_columns.items())},
            # "exist_indexes": [idx.to_dict() for idx in sorted(self.exist_indexes)],
            "basic_cost_explain": self.basic_cost_explain,
            "labels": [idx.to_dict() for idx in self.labels] if self.labels else None,
        }
        if self.label_overrides:
            result["label_overrides"] = {pos: idx.to_dict() for pos, idx in sorted(self.label_overrides.items())}
        if self.basic_cost_real is not None:
            result["basic_cost_real"] = self.basic_cost_real
        return result

    def to_1q(self) -> list["Workload"]:
        """将 Workload 拆分为多个只含单查询的 Workload"""
        workloads = []
        for i, q in enumerate(self.queries):
            w = Workload(
                db=self.db,
                id=self.id * 1000 + i,  # 新的 ID
                queries={q},
                basic_cost_explain=q.basic_cost_explain,
                basic_cost_real=q.basic_cost_real,
                labels=self.labels,
                label_overrides=self.label_overrides,
            )
            workloads.append(w)
        return workloads

    @classmethod
    def from_dict(cls, data: dict):
        """从字典创建 Workload 对象"""
        queries = {QueryInfo.from_dict(q) for q in data["queries"]}
        labels = [Index.from_dict(idx) for idx in data.get("labels", [])] if data.get("labels") else None
        label_overrides = {pos: Index.from_dict(idx) for pos, idx in data.get("label_overrides", {}).items()}
        return cls(
            db=data.get("db", "imdb"),  # 向后兼容. 添加此字段前生成过的数据集均基于 imdb
            id=data.get("id", 0),
            queries=queries,
            basic_cost_explain=data.get("basic_cost_explain", None),
            basic_cost_real=data.get("basic_cost_real"),
            labels=labels,
            label_overrides=label_overrides,
        )


class WorkloadsCompressor:
    def __init__(self, coder: ColumnCoder) -> None:
        self.coder = coder

    def encode_query(self, db: str, q: QueryInfo) -> dict[str, Any]:
        """编码单个 QueryInfo 对象."""

        def ordered(d: dict[str, set[str]]) -> dict[str, list[str]]:
            return {k: sorted(v) for k, v in sorted(d.items())}

        encoded = {
            "db": db,
            "sql": q.sql,
            "tables_columns": self.coder.encode_table_columns(db, ordered(q.tables_columns)),
            "large_columns": self.coder.encode_table_columns(db, ordered(q.large_columns)),
            "exist_indexes": [self.coder.encode_index(db, idx) for idx in sorted(q.exist_indexes)],
            "basic_cost_explain": q.basic_cost_explain,
            "frequency": q.frequency,
        }
        if q.basic_cost_real is not None:
            encoded["basic_cost_real"] = q.basic_cost_real
        return encoded

    def decode_query(self, db: str, data: dict[str, Any]) -> QueryInfo:
        """解码单个 QueryInfo 对象."""

        def collect(columns: list[tuple[str, str, str]]) -> dict[str, set[str]]:
            if not columns:
                return {}
            assert all(db == d for d, _, _ in columns), "列所属数据库不一致"
            result: dict[str, set[str]] = {}
            for _, table, column in columns:
                result.setdefault(table, set()).add(column)
            return result

        tables_columns = collect(self.coder.decode_table_columns(data["tables_columns"]))
        large_columns = collect(self.coder.decode_table_columns(data["large_columns"]))
        exist_indexes = {self.coder.decode_index(idx) for idx in data.get("exist_indexes", [])}
        basic_cost_real = data.get("basic_cost_real", None)

        return QueryInfo(
            sql=data["sql"],
            tables_columns=tables_columns,
            large_columns=large_columns,
            exist_indexes=exist_indexes,
            basic_cost_explain=data.get("basic_cost_explain", 0),
            basic_cost_real=basic_cost_real,
            frequency=data.get("frequency", 1),
        )

    def compress(self, workloads: Sequence[Workload]):
        """压缩 workloads 数据集."""
        sql_codes: dict[tuple[str, int], int] = {}
        sql_count = 0
        sql_infos = []
        for w in workloads:
            for q in sorted(w.queries):
                if (k := (q.sql, q.frequency)) not in sql_codes:
                    sql_codes[k] = sql_count
                    sql_count += 1
                    sql_infos.append(self.encode_query(w.db, q))

        compressed = []
        for w in workloads:
            compressed.append(
                {
                    "db": w.db,
                    "id": w.id,
                    "sqls_hash": w.sqls_hash,
                    "query_codes": [sql_codes[(q.sql, q.frequency)] for q in sorted(w.queries)],
                    "labels": [self.coder.encode_index(w.db, idx) for idx in w.labels] if w.labels else None,
                    "label_overrides": {
                        pos: self.coder.encode_index(w.db, idx) for pos, idx in sorted(w.label_overrides.items())
                    },
                }
            )
        return {
            "column_full_names": self.coder.decode_tbl,
            "sql_infos": sql_infos,
            "workloads": compressed,
        }

    @staticmethod
    def decompress(data: dict[str, Any]) -> list[Workload]:
        assert "column_full_names" in data
        assert "sql_infos" in data
        assert "workloads" in data
        coder = ColumnCoder(column_full_names=data["column_full_names"])
        compressor = WorkloadsCompressor(coder)
        sql_infos = [compressor.decode_query(info["db"], info) for info in data["sql_infos"]]
        workloads = []
        for w_data in data["workloads"]:
            queries = {sql_infos[code] for code in w_data["query_codes"]}
            labels = [coder.decode_index(idx) for idx in w_data.get("labels", [])] if w_data.get("labels") else None
            label_overrides = {pos: coder.decode_index(idx) for pos, idx in w_data.get("label_overrides", {}).items()}
            workloads.append(
                Workload(
                    db=w_data.get("db", "imdb"),
                    id=w_data.get("id", 0),
                    queries=queries,
                    labels=labels,
                    label_overrides=label_overrides,
                )
            )
        return workloads


def generate_workloads(
    db_name: str,
    queries: set[QueryInfo],
    n: int = 100,
    min_size: int = 5,
    max_size: int = 20,
    max_frequency: int = 10,
) -> list[Workload]:
    """
    从所有查询中随机生成工作负载.

    Args:
        db_name: 数据库名称
        queries: 查询列表
        n: 生成的工作负载数量
        min_size: workload 中最小查询数量
        max_size: workload 中最大查询数量
        max_frequency: workload 中查询的最大频率

    Returns:
        工作负载列表
    """
    logger.info(
        f"从 {len(queries)} 条查询中生成 {n} 个工作负载, 每个工作负载包含 {min_size} 到 {max_size} 个查询, 查询最大频率为 {max_frequency}"
    )
    import random

    queries_list = list(queries)
    workloads = []
    for i in range(n):
        size = random.randint(min_size, max_size)
        if min_size == max_size and size == 1:
            # 如果 min_size 和 max_size 相等且都为 1, 即每个 workload 只包含一个查询
            # 则保证每个 workload 的查询不同, 直接按序取
            qs = {queries_list[i].with_frequency(random.randint(1, max_frequency))}
        else:
            qs = {q.with_frequency(random.randint(1, max_frequency)) for q in random.sample(queries_list, size)}
        workloads.append(Workload(db_name, i, qs))

    logger.info(f"共生成 {len(workloads)} 个工作负载")
    return workloads


def show_workloads_statistics(workloads: Sequence[Workload]) -> None:
    """显示workloads的统计信息和分布."""
    if not workloads:
        logger.info("没有workloads数据")
        return

    logger.info("=" * 60)
    logger.info("WORKLOADS 统计信息")
    logger.info("=" * 60)

    # 基本统计
    logger.info(f"总workloads数量: {len(workloads)}")

    # 收集各项指标
    sql_lengths = [w.total_sql_length for w in workloads]
    table_counts = [w.total_table_count for w in workloads]
    column_counts = [w.total_column_count for w in workloads]
    query_counts = [w.total_query_count for w in workloads]
    unique_query_counts = [w.unique_query_count for w in workloads]
    index_counts = [w.total_index_count for w in workloads]
    avg_sql_lengths = [w.avg_sql_length for w in workloads]
    basic_costs = [w.basic_cost for w in workloads]

    # 显示分布统计
    def show_metric_stats(metric_name: str, values: list[int] | list[float]) -> None:
        logger.info(f"\n{metric_name}:")
        logger.info(f"  最小值: {min(values)}")
        logger.info(f"  最大值: {max(values)}")
        logger.info(f"  平均值: {sum(values) / len(values):.2f}")
        logger.info(f"  中位数: {sorted(values)[len(values) // 2]:.2f}")

        # 分布区间统计
        sorted_values = sorted(values)
        p25 = sorted_values[len(values) // 4]
        p50 = sorted_values[len(values) // 2]
        p75 = sorted_values[len(values) * 3 // 4]
        p90 = sorted_values[int(len(values) * 0.9)]
        p95 = sorted_values[int(len(values) * 0.95)]
        logger.info(f"  25%分位数: {p25:.2f}")
        logger.info(f"  50%分位数: {p50:.2f}")
        logger.info(f"  75%分位数: {p75:.2f}")
        logger.info(f"  90%分位数: {p90:.2f}")
        logger.info(f"  95%分位数: {p95:.2f}")

    # show_metric_stats("SQL字符串总长度", sql_lengths)
    show_metric_stats("涉及表数", table_counts)
    show_metric_stats("涉及列数", column_counts)
    show_metric_stats("查询总数量（含频率）", query_counts)
    # show_metric_stats("唯一查询数量", unique_query_counts)
    # show_metric_stats("已存在索引数量", index_counts)
    # show_metric_stats("平均SQL长度", avg_sql_lengths)
    # show_metric_stats("基础成本", basic_costs)

    # 数据库分布
    db_counts = {}
    for w in workloads:
        db_counts[w.db] = db_counts.get(w.db, 0) + 1

    logger.info("\n数据库分布:")
    for db, count in db_counts.items():
        logger.info(f"  {db}: {count} 个workloads ({count / len(workloads) * 100:.1f}%)")

    # 标签统计（如果有的话）
    labeled_count = sum(1 for w in workloads if w.labels)
    if labeled_count > 0:
        logger.info("\n标签统计:")
        logger.info(
            f"  有标签的workloads: {labeled_count}/{len(workloads)} ({labeled_count / len(workloads) * 100:.1f}%)"
        )

        label_counts = [len(w.labels) for w in workloads if w.labels]
        if label_counts:
            show_metric_stats("推荐索引数量", label_counts)

    logger.info("=" * 60)


def save_workloads(p: str | Path, workloads: Sequence[Workload]):
    show_workloads_statistics(workloads)
    all_db = {w.db for w in workloads}
    compressor = WorkloadsCompressor(ColumnCoder.from_schemas((db, schemas[db]) for db in all_db))
    logger.info(f"保存 workloads 到 {str(p)}")
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as file:
        serializable_workloads = compressor.compress(workloads)
        json.dump(serializable_workloads, file, ensure_ascii=False, indent=4)


def load_workloads(path: str | Path, must_contiguous: bool = False) -> list[Workload]:
    """
    从指定路径加载数据集.

    Args:
        must_contiguous: 检查 ID 从 0 开始且连续, 以便可直接通过 ID 索引相应的 workload. 目前已无任何位置有此需要.
    """
    with open(path, "r", encoding="utf-8") as f:
        workloads_data = json.load(f)

    if "column_full_names" in workloads_data and "sql_infos" in workloads_data:
        # 使用 WorkloadsCompressor 解压缩数据
        workloads = WorkloadsCompressor.decompress(workloads_data)
    elif isinstance(workloads_data, list):
        # 使用 from_dict 方法将序列化的数据转换回 Workload 对象
        workloads = [Workload.from_dict(w_data) for w_data in workloads_data]
    else:
        raise ValueError("无法识别的 workloads 数据格式")

    # 检查数据有效性
    assert len(workloads) > 0, "加载的 workloads 数据为空"
    if must_contiguous:
        assert all(w.id == i for i, w in enumerate(workloads)), "Workload ID 不连续或不是从0开始"
    else:
        assert len({w.id for w in workloads}) == len(workloads), "workload ID 不唯一"

    logger.info(f"从 {str(path)} 加载了 {len(workloads)} 个 workloads")

    # 显示workloads统计信息
    show_workloads_statistics(workloads)

    return workloads


def resolve_databases(workloads: Sequence[Workload]) -> set[str]:
    """返回 workloads 中涉及的所有数据库名称."""
    return {w.db for w in workloads}

import re
from typing import Literal

from .ia_logging import logger
from .mm.model import COLUMN_TOKEN, SQL_TOKEN
from .schemas import ColumnMetric, schemas
from .workload import Workload


logger = logger.getChild(__name__)


QUERIES_TITLE = "# queries\n"
WORKLOAD_ID_TITLE = "# workload id\n"
WORKLOAD_ID_RE = re.compile(r"# workload id\n(\d+)")

INSTRUCT1 = (
    "You are an experienced Database Administrator, please recommend indexes for the given SQL queries based on the given metadata to optimize performance.\n"
    "Output recommended indexes one per line in format: table_name(column_name1,...)\n"
    "Do not output any other message.\n"
)


def to_prompt_old(w: Workload, column_mm: bool, sql_mm: bool) -> str:
    """
    生成索引推荐 prompt.

    Args:
        w: Workload 对象
        column_mm: 是否启用 column 多模态
        sql_mm: 是否启用 sql 多模态
    """
    prompt = f"""{INSTRUCT1}
# table and column names in queries
{w.sprint_tables_columns}

{"# large columns (may cause index size exceed limit)" if len(w.large_columns) > 0 else ""}
{w.sprint_large_columns}

# exist indexes
{w.sprint_exist_indexes}

{WORKLOAD_ID_TITLE}{w.id}"""

    # if not sql_mm:  # 如果不启用 sql 多模态, 需向 prompt 添加文本 SQL
    #     prompt += f"\n\n{QUERIES_TITLE}{w.sprint_sqls}\n"
    if sql_mm:  # 启用 sql 多模态, 向 prompt 添加 SQL special token
        prompt += f"\n\n# queries ({len(w.queries)})" + (f"\n{SQL_TOKEN}" * len(w.queries))

    if column_mm:  # 启用 column 多模态, 向 prompt 添加 column special token
        prompt += f"\n\n# columns ({len(w.column_ids)})" + (f"\n{COLUMN_TOKEN}" * len(w.column_ids))

    return prompt


def to_prompt(w: Workload, column_mm: bool, sql_mm: bool, add_plus: bool) -> str:
    column_lines: list[str] = []
    for table in sorted(w.tables_columns):
        column_lines.append(f"## table: {table} {COLUMN_TOKEN}" if column_mm and add_plus else f"## table: {table}")
        for column in sorted(w.tables_columns[table]):
            column_lines.append(f"{column}: {COLUMN_TOKEN}" if column_mm else column)

    prompt = INSTRUCT1 + "\n# table and columns in Workload\n" + "\n".join(column_lines)
    prompt += "\n\n# exist indexes\n" + w.sprint_exist_indexes
    prompt += f"\n\n{WORKLOAD_ID_TITLE}{w.id}"

    if sql_mm:
        prompt += f"\n\n# queries ({len(w.queries)})" + (f"\n{SQL_TOKEN}" * len(w.queries))

    return prompt


def to_prompt1(w: Workload, column_mm: bool, sql_mm: bool) -> str:
    if w.predicate_stats is None:
        raise ValueError("workload.predicate_stats is required to build column statistics prompt")

    schema = schemas[w.db]
    stats = w.predicate_stats
    db_stats = schema.column_stats

    def _cached_stat(table: str, col: str, metric: ColumnMetric) -> float | int | None:
        key = (table, col, metric)
        if key in db_stats:
            return db_stats[key]
        if metric == "row_count" and table in schema.tables:
            pk = schema.tables[table].any_col
            pk_key = (table, pk, metric)
            if pk_key in db_stats:
                return db_stats[pk_key]
        return None

    def _fmt_number(value: float | int | None) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def _average(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    columns: set[tuple[str, str]] = set()
    for table, cols in w.tables_columns.items():
        for col in cols:
            columns.add((table, col))

    def _add_from_keys(keys: set[str]) -> None:
        for key in keys:
            if "." not in key:
                continue
            table, col = key.split(".", 1)
            columns.add((table, col))

    _add_from_keys(set(stats.column_selectivities.keys()))
    _add_from_keys(set(stats.aggregate.join_predicates.keys()))
    _add_from_keys(set(stats.aggregate.group_order_columns.keys()))

    table_columns: dict[str, list[str]] = {}
    for table, col in columns:
        table_columns.setdefault(table, []).append(col)

    column_lines: list[str] = ["col_name,count,NDV,selectivity,join_freq,group_order_freq", ""]
    for table in sorted(table_columns):
        column_lines.append(f"## table: {table}")
        for col in sorted(table_columns[table]):
            key = f"{table}.{col}"
            row_count = _cached_stat(table, col, "row_count")
            ndv = _cached_stat(table, col, "distinct_count")
            selectivity = _average(stats.column_selectivities.get(key, []))
            join_cnt = stats.aggregate.join_predicates.get(key, 0)
            group_cnt = stats.aggregate.group_order_columns.get(key, 0)
            column_lines.append(
                ",".join(
                    [
                        col,
                        _fmt_number(row_count),
                        _fmt_number(ndv),
                        _fmt_number(selectivity),
                        str(join_cnt),
                        str(group_cnt),
                    ]
                )
            )
        column_lines.append("")

    prompt = INSTRUCT1 + "\n# column statistics\n" + "\n".join(column_lines)
    prompt += f"\n\n{WORKLOAD_ID_TITLE}{w.id}"

    # if not sql_mm:
    #     prompt += f"\n\n{QUERIES_TITLE}{w.sprint_sqls}\n"
    if sql_mm:
        prompt += f"\n\n# queries ({len(w.queries)})" + (f"\n{SQL_TOKEN}" * len(w.queries))

    if column_mm:
        prompt += f"\n\n# columns ({len(w.column_ids)})" + (f"\n{COLUMN_TOKEN}" * len(w.column_ids))

    return prompt


def to_data_sample(
    w: Workload, column_mm: bool, sql_mm: bool, prompt_type: Literal["general", "prompt1", "plus"] = "general"
) -> dict:
    """
    生成 LLM 数据集样本.

    Args:
        w: Workload 对象
        column_mm: 是否启用 column 多模态
        sql_mm: 是否启用 sql 多模态
    """
    use_prompt1 = prompt_type == "prompt1"
    add_plus = prompt_type == "plus"
    sample = {
        "input": "",
        "output": w.sprint_labels,
        "instruction": to_prompt1(w, column_mm, sql_mm) if use_prompt1 else to_prompt(w, column_mm, sql_mm, add_plus),
        "columns": w.column_ids_plus if add_plus else w.column_ids,
        "sqls": w.sqls,
    }
    return sample


def continue_prompt(group_size: int = -1, cur_profit: float = 0, cur_size: float = 0) -> str:
    """生成用于继续生成索引的 prompt."""
    # return "Please continue to recommend indexes."
    return f"Please continue to recommend {group_size} indexes."


def get_workload_id(prompt: str) -> int:
    """从 prompt+resp 中提取 workload id."""
    if r := WORKLOAD_ID_RE.search(prompt):
        return int(r.group(1))
    raise ValueError(f"workload id not found in prompt: {prompt}")


def get_continue_count(prompt: str) -> int:
    """获取整轮对话中 user 发出的 continue 指令的数量."""
    return prompt.count("continue to recommend")

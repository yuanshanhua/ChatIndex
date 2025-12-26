from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Literal, cast

import psycopg
import pymysql

from ..ia_logging import logger


if TYPE_CHECKING:
    from ..db import Connection

logger = logger.getChild("schemas")


def literal(x: str) -> Literal[""]:
    """仅用于欺骗 psycopg 的类型检查."""
    return cast(Literal[""], x)


ColumnMetric = Literal[
    "row_count",  # 表的总行数
    "null_count",  # 列中 NULL 的数量
    "not_null_count",  # 列中非 NULL 的数量
    "distinct_count",  # 列中不同取值的数量
    "min",  # 列的最小值
    "max",  # 列的最大值
    "avg",  # 列的平均值 (数值类型)
    "sum",  # 列值求和 (数值类型)
    "max_length",  # 文本列的最大字符长度
    "min_length",  # 文本列的最小字符长度
    "avg_length",  # 文本列的平均字符长度
]


def add_table_stats(schema: "DatabaseSchema", table: str, row_count: int, distinct_counts: Mapping[str, int]):
    """Populate cached ``row_count`` and ``distinct_count`` for a table."""

    for col in schema.tables[table].column_names:
        schema.column_stats[(table, col, "row_count")] = row_count
    for col, value in distinct_counts.items():
        schema.column_stats[(table, col, "distinct_count")] = value


_COLUMN_METRIC_SQL: dict[ColumnMetric, str] = {
    "row_count": "SELECT COUNT(*) FROM {table};",
    "null_count": "SELECT SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) FROM {table};",
    "not_null_count": "SELECT SUM(CASE WHEN {column} IS NOT NULL THEN 1 ELSE 0 END) FROM {table};",
    "distinct_count": "SELECT COUNT(DISTINCT {column}) FROM {table};",
    "min": "SELECT MIN({column}) FROM {table};",
    "max": "SELECT MAX({column}) FROM {table};",
    "avg": "SELECT AVG({column}) FROM {table};",
    "sum": "SELECT SUM({column}) FROM {table};",
    "max_length": "SELECT MAX({column_length}) FROM {table};",
    "min_length": "SELECT MIN({column_length}) FROM {table};",
    "avg_length": "SELECT AVG({column_length}) FROM {table};",
}


@dataclass
class Column:
    name: str
    typ: str
    foreign_key: "Column | None" = None  # 外键列
    table: str = ""  # 所属表名称

    def __hash__(self) -> int:
        return hash((self.name, self.typ))

    @property
    def tb_type(self) -> str:
        """
        将列类型转换为 TaBERT 支持的类型.
        """
        if self.typ == "integer":
            return "real"
        elif self.typ == "character varying":
            return "text"
        else:
            raise ValueError(f"Unsupported column type: {self.typ}")


class TableSchema:
    def __init__(
        self,
        name: str,
        columns: list[Column],
        *,
        primary_key: str | Iterable[str] | None = None,
        large_for_index: list[str] = [],  # text 等大字段不适合作为索引
    ):
        self.name = name
        for col in columns:
            col.table = name
        self.columns = {c.name: c for c in set(columns)}
        self.column_names = {col.name for col in columns}
        self.ordered_column_names = sorted([col.name for col in columns])  # 保持列顺序一致
        self.large_columns = set(large_for_index)
        if primary_key is None:
            primary_keys: tuple[str, ...] = tuple()
        elif isinstance(primary_key, str):
            primary_keys = (primary_key,)
        else:
            primary_keys = tuple(primary_key)
        self.primary_key_columns: tuple[str, ...] = primary_keys
        # 兼容单主键的旧用法; 联合主键时该字段为空
        self.primary_key: str | None = primary_keys[0] if len(primary_keys) == 1 else None
        self.any_col = columns[0].name  # 因为主键可空, 此时需要一个非空列作为 row_count 统计依据

    def __str__(self) -> str:
        return f"{self.name}({','.join(self.column_names)})"

    def select_sql(self, columns: Iterable[str] | None = None, n: int = 1):
        """
        生成 SQL 语句, 从此表中随机获取 n 条记录.
        """
        if columns is None:
            columns = self.ordered_column_names
        return literal(f"SELECT {','.join(columns)} FROM {self.name} ORDER BY random() LIMIT {n};")

    def get_sample_value(self, conn: "Connection", n: int = 1):
        """
        获取表的样本值. 返回值顺序与 ordered_column_names 一致.
        """
        cursor = conn.cursor()
        try:
            cursor.execute(literal(f"SELECT COUNT(*) FROM {self.name};"))
            count = cursor.fetchall()[0][0]
            n = min(n, count)
            cursor.execute(self.select_sql(n=n))
            rows = cursor.fetchall()
            return rows
        except Exception as e:
            raise Exception(f"获取表 {self.name} 的样本值失败: {e}")
        finally:
            cursor.close()


class DatabaseSchema:
    def __init__(
        self,
        name: str,
        *,
        tables: list[TableSchema],
        table_count: int = 0,
        indexes: list["Index"],
        index_count: int = 0,  # 包括主键索引
    ):
        """
        表示一个数据库的模式.

        Args:
            tables: 所有表的模式.
            indexes: 所有非主键索引. 主键索引将自动添加.
        """
        if table_count and len(tables) != table_count:
            raise ValueError(f"数据库 {name} 表数量与预期不符: 预计 {table_count} 实际 {len(tables)}")
        pk_count = len([t for t in tables if t.primary_key_columns])
        if index_count and len(indexes) + pk_count != index_count:
            raise ValueError(f"数据库 {name} 索引数量与预期不符: 预计 {index_count} 实际 {len(indexes) + pk_count}")

        self.name = name
        self.tables = {t.name: t for t in tables}
        self.table_indexes = {t: [index for index in indexes if index.table == t] for t in self.tables}
        self.indexes = set(indexes)
        # 自动添加主键索引
        for table in self.tables.values():
            if not table.primary_key_columns:
                continue
            pindex = Index(table.name, list(table.primary_key_columns))
            if pindex in self.indexes:
                continue
            self.table_indexes[table.name].append(pindex)
            self.indexes.add(pindex)
        # 检查索引有效性
        for index in self.indexes:
            assert index.is_valid(self), f"Invalid index: {index}"
        # relations
        self.foreign_keys: dict[str, list[tuple[str, str, str]]] = {}  # 表名 -> (列名, 外键表名, 外键列名)
        self.column_stats: dict[tuple[str, str, ColumnMetric], int | float | None] = {}

    @cached_property
    def is_columns_unique(self) -> bool:
        """检查此数据库的所有列名是否唯一, 因此可以通过简单的列名匹配检索 SQL 中包含的表和列名."""
        all_columns = set()
        for table in self.tables.values():
            for col in table.columns.values():
                if col.name in all_columns:
                    logger.info(f"列名 {col.name} 在表 {table.name} 中重复, 可能导致匹配错误, 因此不使用简单列名匹配.")
                    return False
                all_columns.add(col.name)
        all_columns = sorted(all_columns, key=lambda x: len(x))
        # 检查是否有列名是其他列名的前缀
        # for i in range(len(all_columns)):
        #     for j in range(i + 1, len(all_columns)):
        #         if all_columns[j].startswith(all_columns[i]):
        #             logger.warning(
        #                 f"列名 {all_columns[i]} 是 {all_columns[j]} 的前缀, 可能导致匹配错误, 因此不使用简单列名匹配."
        #             )
        #             return False
        #         if all_columns[j].endswith(all_columns[i]):
        #             logger.warning(
        #                 f"列名 {all_columns[i]} 是 {all_columns[j]} 的后缀, 可能导致匹配错误, 因此不使用简单列名匹配."
        #             )
        #             return False
        return True

    @cached_property
    def ordered_columns(self) -> list[tuple[str, str]]:
        """返回 (列名,表名) 的列表, 按列名长度从长到短排序."""
        all_columns = []
        for table in self.tables.values():
            for col in table.columns:
                all_columns.append((col, table.name))
        all_columns = sorted(all_columns, key=lambda x: len(x[0]), reverse=True)
        return all_columns

    @classmethod
    def from_connection(cls, name: str, conn: "Connection") -> "DatabaseSchema":
        """
        使用数据库连接查询 schema 信息并创建 DatabaseSchema 对象, 支持 pg 和 mysql.

        注意结果中的 large_columns 仅供参考, 需要人工调整.

        Args:
            name: 数据库名称
            conn: 数据库连接

        Returns:
            DatabaseSchema: 数据库schema对象
        """
        if isinstance(conn, psycopg.Connection):
            return cls._from_pg(name, conn)
        elif isinstance(conn, pymysql.Connection):
            return cls._from_mysql(name, conn)
        else:
            raise ValueError(f"Unsupported connection type: {type(conn)}")

    @classmethod
    def _from_pg(cls, name: str, conn: psycopg.Connection) -> "DatabaseSchema":
        cursor = conn.cursor()
        try:
            # 查询所有表
            tables = []
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            table_names = [row[0] for row in cursor.fetchall()]

            # 查询每个表的列信息
            for table_name in table_names:
                # 获取列信息
                cursor.execute(
                    """
                    SELECT column_name, data_type, character_maximum_length
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position
                """,
                    (table_name,),
                )
                columns_data = cursor.fetchall()
                columns = [Column(name=row[0], typ=row[1]) for row in columns_data]
                column_types = {col.name: col.typ for col in columns}
                column_lengths = {row[0]: row[2] for row in columns_data}

                if not columns:
                    continue

                # 查找主键
                cursor.execute(
                    """
                    SELECT a.attname
                    FROM pg_index i
                    JOIN unnest(i.indkey) WITH ORDINALITY AS idx(attnum, ordinality) ON true
                    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = idx.attnum
                    WHERE i.indrelid = %s::regclass AND i.indisprimary
                    ORDER BY idx.ordinality
                """,
                    (table_name,),
                )
                pk_rows = cursor.fetchall()
                primary_keys: list[str] | None
                if not pk_rows:
                    primary_keys = None  # 无主键
                else:
                    primary_keys = [row[0] for row in pk_rows]

                # 识别大字段类型
                large_columns = []
                for col_name, dtype in column_types.items():
                    max_len = column_lengths.get(col_name)
                    if any(dtype.startswith(t) for t in ["text", "json", "xml", "bytea"]) or (
                        dtype == "character varying" and (max_len is None or max_len > 512)
                    ):
                        large_columns.append(col_name)

                tables.append(
                    TableSchema(
                        name=table_name, columns=columns, primary_key=primary_keys, large_for_index=large_columns
                    )
                )

            # 查询所有非主键索引
            cursor.execute("""
                SELECT
                    t.relname AS table_name,
                    i.relname AS index_name,
                    array_agg(a.attname ORDER BY c.ordinality) AS columns
                FROM
                    pg_index idx
                    JOIN pg_class i ON i.oid = idx.indexrelid
                    JOIN pg_class t ON t.oid = idx.indrelid
                    JOIN pg_namespace n ON n.oid = t.relnamespace
                    JOIN LATERAL unnest(idx.indkey) WITH ORDINALITY AS c(attnum, ordinality) ON true
                    JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = c.attnum
                WHERE
                    n.nspname = 'public'
                    AND NOT idx.indisprimary
                    AND NOT idx.indisunique
                GROUP BY
                    t.relname, i.relname
                ORDER BY
                    t.relname, i.relname
            """)

            indexes = []
            for row in cursor.fetchall():
                table_name, _, columns = row
                indexes.append(Index(table=table_name, columns=columns))

            # 创建DatabaseSchema
            return cls(name=name, tables=tables, indexes=indexes)
        finally:
            cursor.close()

    @classmethod  # todo 检查此方法正确性
    def _from_mysql(cls, name: str, conn: pymysql.Connection) -> "DatabaseSchema":
        cursor = conn.cursor()
        try:
            # 查询所有表
            tables = []
            cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """,
                (name,),
            )
            table_names = [row[0] for row in cursor.fetchall()]

            # 查询每个表的列信息
            for table_name in table_names:
                # 获取列信息
                cursor.execute(
                    """
                    SELECT column_name, data_type, character_maximum_length
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                """,
                    (name, table_name),
                )
                columns_data = cursor.fetchall()
                columns = [Column(name=row[0], typ=row[1]) for row in columns_data]
                column_names = [col.name for col in columns]
                column_types = {col.name: col.typ for col in columns}
                column_lengths = {row[0]: row[2] for row in columns_data}

                if not columns:
                    continue

                # 查找主键
                cursor.execute(
                    """
                    SELECT column_name
                    FROM information_schema.key_column_usage
                    WHERE table_schema = %s AND table_name = %s AND constraint_name = 'PRIMARY'
                    ORDER BY ordinal_position
                    LIMIT 1
                """,
                    (name, table_name),
                )
                primary_key_result = cursor.fetchone()
                primary_key = primary_key_result[0] if primary_key_result else column_names[0]

                # 识别大字段类型
                large_columns = []
                for col_name, dtype in column_types.items():
                    max_len = column_lengths.get(col_name)
                    if any(
                        dtype.lower().startswith(t)
                        for t in ["text", "json", "longtext", "mediumtext", "blob", "longblob", "mediumblob"]
                    ) or (dtype.lower() == "varchar" and (max_len is None or max_len > 512)):
                        large_columns.append(col_name)

                tables.append(
                    TableSchema(
                        name=table_name, columns=columns, primary_key=primary_key, large_for_index=large_columns
                    )
                )

            # 查询所有非主键索引
            cursor.execute(
                """
                SELECT
                    table_name,
                    index_name,
                    GROUP_CONCAT(column_name ORDER BY seq_in_index) as columns
                FROM information_schema.statistics
                WHERE table_schema = %s
                    AND index_name != 'PRIMARY'
                    AND non_unique = 1
                GROUP BY table_name, index_name
                ORDER BY table_name, index_name
            """,
                (name,),
            )

            indexes = []
            for row in cursor.fetchall():
                table_name, index_name, columns_str = row
                if columns_str:
                    columns = [col.strip() for col in columns_str.split(",")]
                    indexes.append(Index(table=table_name, columns=columns))

            # 创建DatabaseSchema
            return cls(name=name, tables=tables, indexes=indexes)
        finally:
            cursor.close()

    def fix_tables_columns(self, tables_columns: dict[str, set[str]]):
        """
        检验 tables_columns 的有效性, 并尝试确定未知列的所属表.

        Args:
            tables_columns (dict[str, set[str]]): 从 SQL 提取所得的表名和列名. 不能确定所属表的列名会被记录在 "unknown" 中.

        Returns:
            dict[str, set[str]]: 处理后的表名和列名.

        Raises:
            AssertionError: 如果表名或列名无效, 或仍有未知列未能确定所属表.
        """
        unknowns = set()
        if "unknown" in tables_columns:
            unknowns = tables_columns.pop("unknown")
        for table, columns in tables_columns.items():
            assert table in self.tables, f"Invalid table name: {table}"
            assert columns.issubset(self.tables[table].column_names), f"Invalid column names for [{table}]: {columns}"
        # 尝试确定未知列的所属表
        for c in unknowns.copy():
            for table in self.tables.values():
                if c in table.column_names:
                    tables_columns.setdefault(table.name, set()).add(c)
                if c in unknowns:  # 有可能已经被删除, 因为一个列名可能出现在多个表中
                    unknowns.remove(c)
        assert len(unknowns) == 0, f"Unknown columns: {unknowns}"
        return tables_columns

    def exist_indexes(self, tables_columns: dict[str, set[str]]) -> set["Index"]:
        """
        返回指定表和列上存在的索引.

        Returns:
            list[Index]: 所有索引.
        """
        indexes = set()
        for table, columns in tables_columns.items():
            table_indexes = self.table_indexes.get(table, [])
            for column in columns:
                for index in table_indexes:
                    if column in index.columns:
                        indexes.add(index)
        return indexes

    def large_columns(self, tables_columns: dict[str, set[str]]) -> dict[str, set[str]]:
        """
        返回指定表和列上的大字段.

        Returns:
            dict[str, set[str]]: 各表上的大字段.
        """
        large_columns = {}
        for table, columns in tables_columns.items():
            table_large_columns = self.tables[table].large_columns
            for column in columns:
                if column in table_large_columns:
                    large_columns.setdefault(table, set()).add(column)
        return large_columns

    def to_python(self) -> str:
        """生成 Python 代码, 可完全重建该对象."""
        # 写入模块文件
        c = ""
        c += f"""from .schema import DatabaseSchema, Index, TableSchema, Column


{self.name}_schema = DatabaseSchema(
    "{self.name}",
    table_count={len(self.tables)},
    tables=[
    """
        # 添加表
        for table in self.tables.values():
            c += "TableSchema("
            c += f'"{table.name}",'
            c += "[\n"
            for col in table.ordered_column_names:
                c += f'    Column(name="{col}", typ="{table.columns[col].typ}"),\n'
            c += "],\n"
            if table.primary_key_columns:
                if len(table.primary_key_columns) == 1:
                    c += f'primary_key="{table.primary_key_columns[0]}",\n'
                else:
                    joined = ", ".join(f'"{pk}"' for pk in table.primary_key_columns)
                    c += f"primary_key=[{joined}],\n"
            if table.large_columns:
                c += "large_for_index=["
                c += ", ".join([f'"{col}"' for col in sorted(table.large_columns)])
                c += "],\n"
            c += "),\n"
        c += "],\n"
        c += f"index_count={len(self.indexes)},\n"
        c += "indexes=[\n"
        # 添加索引
        pk_indexes = {
            Index(table.name, list(table.primary_key_columns))
            for table in self.tables.values()
            if table.primary_key_columns
        }
        for index in sorted(self.indexes - pk_indexes):
            c += f'Index("{index.table}", {index.columns}),\n'
        c += "],\n"
        c += ")\n"
        return c

    def add_foreign_key(self, table: str, column: str, foreign_table: str, foreign_column: str):
        """
        添加外键约束.
        """
        if table not in self.tables:
            raise ValueError(f"Invalid table name: {table}")
        if column not in self.tables[table].column_names:
            raise ValueError(f"Invalid column name: {column}")
        if foreign_table not in self.tables:
            raise ValueError(f"Invalid foreign table name: {foreign_table}")
        if foreign_column not in self.tables[foreign_table].column_names:
            raise ValueError(f"Invalid foreign column name: {foreign_column}")
        self.foreign_keys.setdefault(table, []).append((column, foreign_table, foreign_column))
        self.tables[table].columns[column].foreign_key = self.tables[foreign_table].columns[foreign_column]

    def sample_large_columns(self, conn: "Connection", sample_count: int = 5, max_display_length: int = 100):
        """
        从每个表的 large_columns 中采样若干示例数据，用于人工判断是否真的是大字段.

        Args:
            conn: 数据库连接
            sample_count: 每个大字段采样的记录数量，默认为5
            max_display_length: 显示内容的最大长度，超过部分会被截断，默认为100

        Returns:
            None: 直接打印采样结果
        """
        cursor = conn.cursor()
        try:
            print(f"=== 数据库 {self.name} 大字段采样结果 ===\n")

            for table_name, table_schema in self.tables.items():
                if not table_schema.large_columns:
                    continue

                print(f"表: {table_name}")
                print("=" * 50)

                # 检查表是否有数据
                cursor.execute(literal(f"SELECT COUNT(*) FROM {table_name};"))
                count_result = cursor.fetchone()
                if count_result is None:
                    print(f"  表 {table_name} 查询失败\n")
                    continue
                row_count = count_result[0]

                if row_count == 0:
                    print(f"  表 {table_name} 无数据\n")
                    continue

                # 采样每个大字段
                for column_name in sorted(table_schema.large_columns):
                    stats_summary = self._column_reference_stats_summary(conn, table_name, column_name, row_count)
                    header = f"  列: {column_name} (类型: {table_schema.columns[column_name].typ})"
                    if stats_summary:
                        header += f" [{stats_summary}]"
                    print(header)
                    print("  " + "-" * 40)

                    # 获取采样数据，过滤空值
                    actual_sample_count = min(sample_count, row_count)
                    cursor.execute(
                        literal(f"""
                        SELECT {column_name}
                        FROM {table_name}
                        WHERE {column_name} IS NOT NULL
                        ORDER BY random()
                        LIMIT {actual_sample_count};
                        """)
                    )

                    samples = cursor.fetchall()

                    if not samples:
                        print("    无非空数据\n")
                        continue

                    for i, (value,) in enumerate(samples, 1):
                        if value is None:
                            display_value = "NULL"
                        else:
                            value_str = str(value)
                            if len(value_str) > max_display_length:
                                display_value = value_str[:max_display_length] + "..."
                            else:
                                display_value = value_str

                            # 显示字符长度信息
                            length_info = f" (长度: {len(str(value))})"
                            display_value += length_info

                        print(f"    样本 {i}: {display_value}")

                    print()  # 列之间的空行

                print()  # 表之间的空行

        except Exception as e:
            logger.error(f"采样大字段时发生错误: {e}")
            raise
        finally:
            cursor.close()

    def column_statistic(
        self, conn: "Connection", table: str, column: str, metric: ColumnMetric
    ) -> int | float | None:
        """获取指定列的统计信息, 结果会被缓存以避免重复查询."""
        if table not in self.tables:
            raise ValueError(f"Invalid table name: {table}")
        if column not in self.tables[table].column_names:
            raise ValueError(f"Invalid column name: {table}.{column}")
        cache_key = (table, column, metric)
        if cache_key in self.column_stats:
            return self.column_stats[cache_key]

        sql_template = _COLUMN_METRIC_SQL.get(metric)
        if sql_template is None:
            raise ValueError(f"Unsupported metric: {metric}")
        sql = sql_template.format(
            table=table,
            column=column,
            column_length=self._column_length_expr(conn, column),
        )

        cursor = conn.cursor()
        try:
            cursor.execute(literal(sql))
            row = cursor.fetchone()
            value = row[0] if row else None
            if isinstance(value, Decimal):
                value = float(value)
        except Exception as exc:
            logger.error(f"查询 {table}.{column} 的统计信息失败: {metric}: {exc}")
            raise
        finally:
            cursor.close()

        self.column_stats[cache_key] = value
        return value

    def _column_reference_stats_summary(self, conn: "Connection", table: str, column: str, table_rows: int) -> str:
        stats: list[tuple[str, int | float | None]] = [("rows", table_rows)]
        not_null = self.column_statistic(conn, table, column, "not_null_count")
        not_null_value = self._normalize_stat_value(not_null)
        stats.append(("not_null", not_null_value))

        not_null_int = self._stat_as_int(not_null_value)
        if not_null_int is not None:
            stats.append(("null", max(table_rows - not_null_int, 0)))

        distinct = self.column_statistic(conn, table, column, "distinct_count")
        stats.append(("distinct", self._normalize_stat_value(distinct)))

        if self._column_supports_length_metrics(table, column):
            stats.extend(
                [
                    ("len_max", self._normalize_stat_value(self.column_statistic(conn, table, column, "max_length"))),
                    ("len_min", self._normalize_stat_value(self.column_statistic(conn, table, column, "min_length"))),
                    ("len_avg", self._normalize_stat_value(self.column_statistic(conn, table, column, "avg_length"))),
                ]
            )

        parts = []
        for label, value in stats:
            if value is None:
                continue
            parts.append(f"{label}={self._format_stat_value(value)}")
        return ", ".join(parts)

    def _column_supports_length_metrics(self, table: str, column: str) -> bool:
        col_type = self.tables[table].columns[column].typ.lower()
        return "char" in col_type or "text" in col_type

    def _column_length_expr(self, conn: "Connection", column: str) -> str:
        if isinstance(conn, psycopg.Connection):
            return f"char_length({column}::text)"
        if isinstance(conn, pymysql.Connection):
            return f"char_length(CAST({column} AS CHAR))"
        return f"char_length({column})"

    @staticmethod
    def _normalize_stat_value(value: int | float | None) -> int | float | None:
        if value is None:
            return None
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    @staticmethod
    def _stat_as_int(value: int | float | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, float):
            return int(round(value))
        return int(value)

    @staticmethod
    def _format_stat_value(value: int | float) -> str:
        if isinstance(value, float) and not value.is_integer():
            return f"{value:.2f}"
        return str(int(value)) if isinstance(value, float) else str(value)


class Index:
    def __init__(self, table: str, columns: list[str]):
        self.table = table
        self.columns = columns
        self.columns_set = set(columns)
        self.name = f"{self.table}_{'_'.join(self.columns)}_idx"

    def to_dict(self) -> dict:
        return {"table": self.table, "columns": self.columns}

    @classmethod
    def from_dict(cls, data: dict) -> "Index":
        return cls(table=data["table"], columns=data["columns"])

    def is_valid(self, schema: DatabaseSchema) -> bool:
        """
        根据数据库 schema 检查索引是否有效. 仅检查表和列是否存在.
        """
        return self.table in schema.tables and self.columns_set.issubset(schema.tables[self.table].column_names)

    def is_large(self, schema: DatabaseSchema) -> bool:
        """
        检查索引是否包含过大的列.
        """
        return len(self.columns_set.intersection(schema.tables[self.table].large_columns)) > 0

    def __eq__(self, other) -> bool:
        """
        通过比较索引名称来判断两个索引是否相等。

        Args:
            other: 另一个索引对象

        Returns:
            bool: 如果索引名称相同，则返回True，否则返回False
        """
        if not isinstance(other, Index):
            return False
        return self.name == other.name

    def __lt__(self, other: "Index") -> bool:
        return self.name < other.name

    def __hash__(self) -> int:
        """
        返回索引名称的哈希值，使索引可以用于集合操作。

        Returns:
            int: 索引名称的哈希值
        """
        return hash(self.name)

    def __repr__(self) -> str:
        return f"Index({self.table}, {self.columns})"

    def __str__(self) -> str:
        return f"{self.table}({','.join(self.columns)})"

    def is_permutation(self, other: "Index") -> bool:
        """
        检查两个索引是否是相同列的排列组合. 如 (a,b) 和 (b,a).
        """
        return self.table == other.table and set(self.columns) == set(other.columns)

    def is_prefix(self, other: "Index") -> bool:
        """
        检查一个索引是否是另一个索引的前缀. 如 (a,b) 是 (a,b,c) 的前缀.
        """
        if len(self.columns) >= len(other.columns):
            self, other = other, self  # 保证 self 是较短的索引
        return self.table == other.table and other.columns[: len(self.columns)] == self.columns

    def is_subset(self, other: "Index") -> bool:
        """
        检查一个索引是否是另一个索引的子集. 如 (a,c) 是 (a,b,c) 的子集.
        """
        if len(self.columns) >= len(other.columns):
            self, other = other, self
        return self.table == other.table and self.columns_set.issubset(other.columns_set)

    def create_sql(
        self,
        type: Literal["btree", "hash", "gist", "spgist", "gin", "brin"] = "btree",
    ):
        return literal(
            f"CREATE INDEX IF NOT EXISTS {self.name} ON {self.table} USING {type} ({','.join(self.columns)});"
        )

    def drop_sql(self):
        return literal(f"DROP INDEX IF EXISTS {self.name};")

    def create_hypopg_sql(
        self,
        type: Literal["btree", "hash", "brin"] = "btree",
    ):
        return literal(f"SELECT hypopg_create_index('{self.create_sql(type)}');")


class ColumnCoder:
    def __init__(self, column_full_names: Iterable[str]):
        self.decode_tbl = sorted(column_full_names)
        self.encode_tbl = {name: i for i, name in enumerate(self.decode_tbl)}

    @classmethod
    def from_schemas(cls, schemas: Iterable[tuple[str, DatabaseSchema]]) -> "ColumnCoder":
        column_full_names: set[str] = set()
        for db, s in schemas:
            for table in s.tables.values():
                for col in table.columns.values():
                    col_id = f"{db}.{table.name}.{col.name}"
                    if col_id in column_full_names:
                        raise ValueError(f"Column full id conflict: {col_id}")
                    column_full_names.add(col_id)
        return cls(column_full_names)

    def encode_table_columns(
        self, db: str, table_columns: Mapping[str, Iterable[str]], schema: DatabaseSchema | None = None
    ) -> list[int]:
        """
        将同一数据库中的多个列编码为有序整数列表.

        Args:
            db (str): 数据库名称
            table_columns (Mapping[str, Iterable[str]]): 表名称到列名称列表的映射
            schema (DatabaseSchema|None): 可选, 若提供将进行验证

        Returns:
            list[int]: 编码后的整数列表
        """
        if schema:
            for table, columns in table_columns.items():
                assert table in schema.tables, f"Invalid table name: {table}"
                for col in columns:
                    assert col in schema.tables[table].column_names, f"Invalid column name: {col} in table {table}"
        encoded = []
        for table, columns in table_columns.items():
            for col in columns:
                full_name = f"{db}.{table}.{col}"
                code = self.encode_tbl[full_name]
                encoded.append(code)
        return encoded

    def decode_table_columns(self, codes: list[int]) -> list[tuple[str, str, str]]:
        """
        将编码的列整数列表解码为 (数据库名称, 表名称, 列名称) 列表.

        Args:
            codes (list[int]): 编码的整数列表

        Returns:
            list[tuple[str, str, str]]: 解码后的 (数据库名称, 表名称, 列名称) 列表
        """
        decoded = []
        for code in codes:
            full_name = self.decode_tbl[code]
            db, table, col = full_name.split(".", 2)
            decoded.append((db, table, col))
        return decoded

    def encode_index(self, db: str, index: Index, schema: DatabaseSchema | None = None) -> list[int]:
        return self.encode_table_columns(db, {index.table: index.columns}, schema)

    def decode_index(self, codes: list[int]) -> Index:
        """将编码的索引表示解码为 Index 对象. 若不是同一数据库同一表的列将引发异常."""
        decoded = self.decode_table_columns(codes)
        db = decoded[0][0]
        table = decoded[0][1]
        assert all(db == d and table == t for d, t, _ in decoded), "Decoded index columns belong to different tables"
        columns = [col for _, _, col in decoded]
        return Index(table, columns)


def check_redundent_indexes(indexes: list[Index]):
    """
    检查一批索引中的冗余, 如置换, 前缀等.
    """
    permutation: list[tuple[Index, Index]] = []
    prefix: list[tuple[Index, Index]] = []
    for i in range(len(indexes)):
        for j in range(i + 1, len(indexes)):
            cur, oth = indexes[i], indexes[j]
            if cur.is_permutation(oth):
                permutation.append((cur, oth))
            elif cur.is_prefix(oth):
                prefix.append((cur, oth))
    return permutation, prefix


def parse_indexes_str(
    s: str, db: DatabaseSchema | None = None, keep_exist: bool = False
) -> tuple[list["Index"], tuple[int, int, int]]:
    """
    从字符串中提取符合数据库架构的索引.

    Args:
        s (str): 多行字符串, 每行表示一个索引, 格式为 table_name(column_name1 [,...]).
        schema (Optional[DatabaseSchema]): 数据库 schema, 若提供则仅返回非主键的有效索引.
        keep_exist (bool): 是否保留数据库中已存在的索引.

    Returns:
        list[Index]: 消息中包含的所有索引, 保证均有效且不重复.
        tuple[int, int, int]: 三个整数元组, 分别表示: 无效索引数量, 已存在索引数量, 有效但重复索引数量.
    """
    # get model response
    lines = [line.strip() for line in s.split("\n") if line.strip()]
    # get indexes
    indexes = []
    invalid_count = 0
    exist_count = 0
    dup_count = 0
    for line in lines:
        # 检查是否符合 table_name(column_name1 [,...]) 格式. 允许右括号后有一些其他字符, 会被忽略
        if line.count("(") != 1 or line.count(")") != 1:
            invalid_count += 1
            logger.warning(f"索引格式无效: {line}")
            continue
        # if not line.endswith(")"):
        #     invalid_count += 1
        table, column_names = line[: line.rindex(")")].split("(")
        columns = [col.strip() for col in column_names.split(",") if col.strip()]
        if len(columns) == 0:
            invalid_count += 1
            logger.warning(f"索引列为空: {line}")
            continue
        if len(set(columns)) != len(columns):
            invalid_count += 1
            logger.warning(f"索引存在重复列: {line}")
            continue
        index = Index(table.strip(), columns)
        if db:
            if not index.is_valid(db):
                invalid_count += 1
                logger.warning(f"索引不符合schema: {index}")
                continue
            if index in db.indexes:
                if keep_exist:
                    indexes.append(index)
                exist_count += 1
                logger.warning(f"索引已存在: {index}")
                continue
            if index.is_large(db):
                invalid_count += 1
                logger.warning(f"索引体积过大: {index}")
                continue
        if index in indexes:
            logger.warning(f"重复索引: {index}")
            dup_count += 1
            continue
        indexes.append(index)
    return indexes, (invalid_count, exist_count, dup_count)

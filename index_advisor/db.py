import argparse
import time
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from functools import cached_property
from random import random
from typing import Any, Iterable, Literal

import psycopg
import pymysql
import pymysql.cursors
import sqlglot
import sqlglot.expressions
from psycopg.errors import DeadlockDetected, ProgramLimitExceeded
from pymysql.err import OperationalError

from .ia_logging import logger
from .schemas import DatabaseSchema, Index, aliases, schemas
from .schemas.schema import literal
from .workload import QueryInfo


logger = logger.getChild(__name__)

Connection = psycopg.Connection | pymysql.Connection
Cursor = psycopg.Cursor | pymysql.cursors.Cursor

# Multiprocessing helpers for per-process DBOption / connections
_worker_opt_shared: "DBOption | None" = None
_worker_connections_shared: dict[str, Connection] | None = None


@dataclass
class DBOption:
    db_type: Literal["pg", "mysql"]
    databases: list[str]
    user: str
    password: str
    host: str
    port: int
    _raise_exception: bool = True  # schema 不存在时是否抛出异常, 仅用于初始化

    def __post_init__(self):
        # 向后兼容仅传入单个数据库名的情况
        self.db = self.databases[0]
        if self.databases[0] not in schemas:
            if self._raise_exception:
                raise ValueError(f"Error: 未知数据库 [{self.databases[0]}]. 可用: {', '.join(schemas)}")
        else:
            self.schema = schemas[self.databases[0]]

    @cached_property
    def conn(self) -> Connection:
        return self.connect(self.db)

    @cached_property
    def connections(self) -> dict[str, Connection]:
        connections: dict[str, Connection] = {}
        for db in self.databases:
            connections[db] = self.connect(db)
        # 添加别名连接
        for name, anames in aliases.items():
            for aname in anames:
                if name in connections:
                    connections[aname] = connections[name]

        return connections

    def connect(self, db: str):
        if db not in aliases:
            for name, anames in aliases.items():
                if db in anames:
                    db = name
                    break
            else:
                if db not in schemas:
                    raise ValueError(f"Error: 未知数据库 [{db}]. 可用: {', '.join(schemas)}")
        if self.db_type == "mysql":
            return pymysql.connections.Connection(
                host=self.host,
                user=self.user,
                password=self.password,
                database=db,
                port=self.port,
                connect_timeout=10,
                autocommit=False,
            )
        elif self.db_type == "pg":
            return psycopg.connect(self.pg_connect_str(db))
        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}")

    @classmethod
    def add_arg_group_to(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        dbargs = parser.add_argument_group("Database", "数据库选项")
        dbargs.add_argument("--db", nargs="+", default=None, help="数据库名称, 使用空格分隔以指定多个")
        dbargs.add_argument("--dbtype", choices=["pg", "mysql"], default="pg", help="数据库类型")
        dbargs.add_argument("--dbuser", default="postgres", help="数据库用户名")
        dbargs.add_argument("--dbpswd", default="", help="数据库密码")
        dbargs.add_argument("--dbhost", default="localhost", help="数据库 host")
        dbargs.add_argument("--dbport", type=int, default=5432, help="数据库端口号")
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace, raise_exception: bool = True):
        return cls(
            databases=args.db,
            db_type=args.dbtype,
            user=args.dbuser,
            password=args.dbpswd,
            host=args.dbhost,
            port=args.dbport,
            _raise_exception=raise_exception,
        )

    def to_dict(self):
        return {
            "databases": self.databases,
            "db_type": self.db_type,
            "user": self.user,
            "password": self.password,
            "host": self.host,
            "port": self.port,
            "_raise_exception": self._raise_exception,
        }

    @classmethod
    def from_dict(cls, d: dict):
        if "db" in d:
            databases = d["db"] if isinstance(d["db"], list) else [d["db"]]
        else:
            databases = d.get("databases", ["imdb"])
        return cls(
            databases=databases,
            db_type=d.get("db_type", "pg"),
            user=d.get("user", "postgres"),
            password=d.get("password", "123456"),
            host=d.get("host", "localhost"),
            port=d.get("port", 5432),
            _raise_exception=d.get("_raise_exception", True),
        )

    def pg_connect_str(self, db: str) -> str:
        """
        生成 pg 连接字符串.
        Args:
            db (str): 数据库名称
        Returns:
            str: 连接字符串
        """
        s = f"dbname={db} user={self.user} host={self.host} connect_timeout=10"
        if self.port:
            s += f" port={self.port}"
        if self.password:
            s += f" password={self.password}"
        logger.debug(f"pg 连接字符串: {s}")
        return s


def init_worker_connections(opt_dict: Mapping[str, Any]) -> None:
    """Initialize per-process DBOption and connection cache for multiprocessing workers."""

    global _worker_opt_shared, _worker_connections_shared
    _worker_opt_shared = DBOption.from_dict(dict(opt_dict))
    _worker_connections_shared = {}


def worker_connection(db: str) -> Connection:
    """Get or create a cached connection for the given db in the current process."""

    global _worker_connections_shared
    if _worker_opt_shared is None:
        raise RuntimeError("worker connections not initialized")
    if _worker_connections_shared is None:
        _worker_connections_shared = {}
    if db not in _worker_connections_shared:
        _worker_connections_shared[db] = _worker_opt_shared.connect(db)
    return _worker_connections_shared[db]


def handle_sql(sql: str, db: DatabaseSchema):
    """
    根据数据库 schema 处理 SQL 语句.

    Returns:
        1. SQL 中出现的表名和列名, 不能确定所属表的列名会被记录在 "unknown" 中
        3. 大字段及其所属表
        4. 数据库中已存在的索引
    """
    if db.is_columns_unique:
        tables_columns = get_table_columns_simple(sql, db)
    else:
        _, tables_columns = _parse_and_clean_sql(sql)
    tables_columns = db.fix_tables_columns(tables_columns)
    return (tables_columns, db.large_columns(tables_columns), db.exist_indexes(tables_columns))


def get_table_columns_simple(sql: str, db: DatabaseSchema) -> dict[str, set[str]]:
    """
    通过简单的查找列名获取 SQL 中的表名和列名. 按从长到短的顺序查找列名, 这样只需保证不存在完全相同的列名即可.

    Args:
        sql (str): SQL 查询语句
        db (DatabaseSchema): 数据库 schema
    Returns:
        dict[str, set[str]]: 表名 -> 列名集合
    """
    sql = sql.lower()
    tables_columns: dict[str, set[str]] = {}
    for cname, tname in db.ordered_columns:
        if cname in sql:
            sql = sql.replace(cname, " ")  # 避免后续误匹配以 cname 子串为名的列
            tables_columns.setdefault(tname, set()).add(cname)
    return tables_columns


def _parse_and_clean_sql(sql: str) -> tuple[str, dict[str, set[str]]]:
    """
    解析 SQL 语句, 消除其中的别名, 提取其中的表名和列名.

    消除别名不总是安全的, 当存在自连接时, 无法避免别名.
    Args:
        sql (str): sql query

    Returns:
        str: 消除别名后的 sql.
        dict[str, set[str]]: SQL 中出现的表名和列名. 不能确定所属表的列名会被记录在 "unknown" 中.
    """
    exp = sqlglot.parse_one(sql)
    res = {"unknown": set()}
    # 消除表别名定义 t AS title -> title
    alias_to_table = {}
    for e in exp.find_all(sqlglot.expressions.Table):
        res[e.name] = set()  # e.name 必然不是表别名
        a = e.args.get("alias")
        if isinstance(a, sqlglot.expressions.TableAlias):
            if a.name in alias_to_table and alias_to_table[a.name] != e.name:
                raise ValueError(f"同一表别名被多次定义: {a.name}")
            alias_to_table[a.name] = e.name
            a.pop()
    # print(f"alias_to_table: {alias_to_table}")
    # 替换表别名 t.id -> title.id
    for e in exp.find_all(sqlglot.expressions.Column):
        t = e.args.get("table")
        if isinstance(t, sqlglot.expressions.Identifier):
            if t.name in alias_to_table:
                t.args["this"] = alias_to_table[t.name]
    # 消除列别名定义 title.id AS title_id -> title.id
    alias_to_column = {}
    for e in exp.find_all(sqlglot.expressions.Alias):
        c = e.this  # c 是别名指代的原始列
        # print(f"e.this: {c}")
        if not isinstance(c, sqlglot.expressions.Column):
            # 非列别名. 如 count(*) as count
            continue
        # print(f"e.args: {e.args}")
        a = e.args.get("alias")
        assert isinstance(a, sqlglot.expressions.Identifier)
        if a.name in alias_to_column and alias_to_column[a.name] != c:
            raise ValueError(f"同一列别名被多次定义: {a.name}")
        alias_to_column[a.name] = c
        a.pop()
    # print(f"alias_to_column: {alias_to_column}")
    # 替换列别名 title_id -> title.id
    for e in exp.find_all(sqlglot.expressions.Column):
        if not e.table and e.name in alias_to_column:  # e 是别名引用
            c = alias_to_column[e.name]  # c 是别名指代的原始列
            e.replace(c)
    # 消除所有别名后, 记录表名和列名
    for e in exp.find_all(sqlglot.expressions.Column):
        if e.table:  # 以 t.c 出现, 记录为 t 的列
            res[e.table].add(e.name)
        else:  # 仅列名, 记录为未知列
            res["unknown"].add(e.name)
    return str(exp), res


def explain_query(
    cursor: Cursor,
    query: str,
    analyze: bool = False,
    raise_exception: bool = False,
) -> float:
    if isinstance(cursor, psycopg.Cursor):
        return _explain_query_pg(cursor, query, analyze, raise_exception)
    elif isinstance(cursor, pymysql.cursors.Cursor):
        return _explain_query_mysql(cursor, query, analyze, raise_exception)
    else:
        raise ValueError(f"Unsupported cursor type: {type(cursor)}")


def _explain_query_pg(cursor: psycopg.Cursor, query: str, analyze: bool, raise_exception: bool) -> float:
    try:
        cursor.execute(literal(f"EXPLAIN ({'ANALYZE,' if analyze else ''} FORMAT JSON) {query}"))
        res = cursor.fetchall()
        # print(f"Query plan: {res}")
        r = res[0][0][0]["Plan"]
        return r["Actual Total Time"] if analyze else r["Total Cost"]
    except Exception as e:
        if raise_exception:
            raise e
        logger.error(f"explain 失败: {str(e)} Query: {query}")
        return -1


def _explain_query_mysql(cursor: pymysql.cursors.Cursor, query: str, analyze: bool, raise_exception: bool) -> float:
    try:
        cursor.execute(f"EXPLAIN {query}")
        res = cursor.fetchall()
        # print(f"Query plan: {res}")
        # todo 验证正确性
        total_rows = sum(row[0] for row in res)
        return float(total_rows)
    except Exception as e:
        if raise_exception:
            raise e
        logger.error(f"explain 失败: {str(e)} Query: {query}")
        return -1


CostCacheKey = tuple[str, frozenset[Index], bool, bool]
SizeCacheKey = tuple[str, Index]


class CostCache:
    """Cache for query costs with different index configurations"""

    def __init__(self, backing: MutableMapping[CostCacheKey, float] | None = None):
        if backing is None:
            backing = {}
        self._cache: MutableMapping[CostCacheKey, float] = backing

    def get(self, key: CostCacheKey) -> float | None:
        """Get cached cost if exists"""
        return self._cache.get(key)

    def update(self, key: CostCacheKey, cost: float) -> float:
        """Update or add a cost to the cache"""
        self._cache[key] = cost
        return cost

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: CostCacheKey) -> bool:
        return key in self._cache

    def clear(self) -> None:
        """Clear the entire cache"""
        self._cache.clear()

    @staticmethod
    def hypo_key(sql: str, indexes: Iterable[Index]) -> CostCacheKey:
        """Generate a cache key for hypothetical indexes"""
        return (sql, frozenset(indexes), True, False)

    @staticmethod
    def real_key(sql: str, indexes: Iterable[Index], execute: bool) -> CostCacheKey:
        """Generate a cache key for real indexes"""
        return (sql, frozenset(indexes), False, execute)


class SizeCache:
    """Cache for query costs with different index configurations"""

    def __init__(self):
        self._cache: dict[SizeCacheKey, float] = {}

    def get(self, db: str, idx: Index) -> float | None:
        """Get cached cost if exists"""
        return self._cache.get((db, idx))

    def update(self, db: str, idx: Index, cost: float) -> float:
        """Update or add a cost to the cache"""
        self._cache[(db, idx)] = cost
        return cost

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: SizeCacheKey) -> bool:
        return key in self._cache

    def clear(self) -> None:
        """Clear the entire cache"""
        self._cache.clear()


cost_cache = CostCache()
size_cache = SizeCache()


def get_hypo_index_size(cursor: Cursor, db: str, index: Index, cache: SizeCache = size_cache) -> float:
    """估计索引大小, 单位为 Bytes. 首先创建虚拟索引, 而后使用 hypopg_relation_size 获取估计大小, 最后移除此索引."""
    if (s := cache.get(db, index)) is not None:
        return s
    assert isinstance(cursor, psycopg.Cursor), "hypopg 仅支持 PostgreSQL"
    cursor.execute(index.create_hypopg_sql())
    res = cursor.fetchone()
    assert res is not None, "hypopg_create_index 未返回结果"
    oid = res[0][0]
    cursor.execute(literal(f"SELECT * FROM hypopg_relation_size({oid});"))
    res = cursor.fetchone()
    assert res is not None, "hypopg_relation_size 未返回结果"
    size = res[0]
    drop_hypo_index(cursor, oid)
    r = float(size)
    cache.update(db, index, r)
    return r


def create_index_and_get_cost(
    conn: Connection,
    indexes: Iterable["Index"],
    queries: set[QueryInfo],
    execute: bool = False,  # 是否实际执行查询
    cache: CostCache = cost_cache,
) -> float:
    cursor = None
    cost = 0.0
    # 检查缓存
    not_hit: list[tuple[CostCacheKey, QueryInfo]] = []
    for q in queries:
        key = CostCache.real_key(q.sql, indexes, execute)
        if (c := cache.get(key)) is not None:
            # logger.debug(f"查询 {q} 命中缓存")
            cost += c * q.frequency
        else:
            not_hit.append((key, q))
    if not not_hit:
        return cost
    try:
        cursor = conn.cursor()
        # 创建索引（事务内）
        for index in indexes:
            try:
                cursor.execute(index.create_sql())
                logger.debug(f"事务内创建索引 {index.name}")
            except ProgramLimitExceeded as e:
                # index row size ... exceeds btree version 4 maximum 2704 for index ...
                # index row requires ... bytes, maximum size is 8191
                logger.error(f"索引大小超过限制: {indexes} {e}")
                continue
        return -1
        # 执行 EXPLAIN 分析（事务内可见索引）
        for key, q in not_hit:
            cost += cache.update(key, explain_query(cursor, q.sql, execute) * q.frequency)
        return cost
    except DeadlockDetected:
        logger.error("发生死锁, 进行重试")
        conn.rollback()
        time.sleep(random() * 10)
        return create_index_and_get_cost(conn, indexes, queries, execute)
    except OperationalError as e:
        # could not serialize access due to concurrent update
        if "could not serialize access due to concurrent update" in str(e):
            logger.error("发生并发更新冲突, 进行重试")
            conn.rollback()
            time.sleep(random() * 10)
            return create_index_and_get_cost(conn, indexes, queries, execute)
        logger.error(f"操作失败: 数据库错误 {type(e)} {str(e)}")
        return -1
    except Exception as e:
        logger.error(f"操作失败: 意料之外的错误 {type(e)} {str(e)}")
        return -1
    finally:
        # 主动回滚撤销所有 DDL 操作（包括 CREATE INDEX）
        conn.rollback()
        if cursor:
            cursor.close()


def create_hypo_index_and_get_cost(
    conn: Connection,
    indexes: Iterable["Index"],
    queries: set[QueryInfo],
    cost_cache: CostCache = cost_cache,
) -> float:
    cursor = None
    cost = 0.0
    # 检查缓存
    not_hit: list[tuple[CostCacheKey, QueryInfo]] = []
    for q in queries:
        key = CostCache.hypo_key(q.sql, indexes)
        if (c := cost_cache.get(key)) is not None:
            # logger.debug(f"查询 {q} 命中缓存")
            cost += c * q.frequency
        else:
            not_hit.append((key, q))
    if not not_hit:
        return cost
    try:
        cursor = conn.cursor()
        # 清空假设索引
        cursor.execute("SELECT hypopg_reset();")
        # 创建假设索引
        for index in indexes:
            create_sql = index.create_hypopg_sql()
            cursor.execute(create_sql)
            logger.debug(f"创建假设索引: {index.name}")

        for key, q in not_hit:
            cost += cost_cache.update(key, explain_query(cursor, q.sql) * q.frequency)
        return cost

    except Exception as e:
        logger.error(f"操作失败: {str(e)}")
        return -1
    finally:
        conn.rollback()
        if cursor:
            cursor.close()


@dataclass
class IndexInfo:
    index: Index
    cost: float
    profit: float  # 相对于前一个索引的收益, 即前一个 cost - 当前 cost


@dataclass
class IdxSeqInfo:
    basic_cost: float
    infos: list[IndexInfo]

    @property
    def indexed_cost(self) -> float:
        if not self.infos:
            return self.basic_cost
        return self.infos[-1].cost


def create_hypo_index_and_get_all_info(
    conn: Connection,
    indexes: Iterable["Index"],
    queries: set[QueryInfo],
    cost_cache: CostCache = cost_cache,
) -> IdxSeqInfo:
    """
    获取索引序列中每个前缀的 cost 及无索引 cost. 返回值 IdxSeqInfo.infos 与 indexes 一一对应.

    这意味着依次创建 indexes 中的索引并获取相应的 cost.
    当需要 indexes 的任意前缀索引序列的 cost 时, 使用此方法相比于多次调用
    create_hypo_index_and_get_cost 复杂度从 O(n^2) 降为 O(n).
    当只需要最终 cost 时, 使用 create_hypo_index_and_get_cost 更加高效.

    无缓存时, 此函数实际调用 explain 次数为 len(queries) * (len(indexes)+1).
    """
    cursor = None
    indexes = list(indexes)
    try:
        cursor = conn.cursor()
        # 清空假设索引
        cursor.execute("SELECT hypopg_reset();")
        # 获取基本 cost
        base_cost = 0.0
        for q in queries:
            key = CostCache.hypo_key(q.sql, [])
            if (c := cost_cache.get(key)) is not None:
                base_cost += c * q.frequency
            else:
                base_cost += cost_cache.update(key, explain_query(cursor, q.sql) * q.frequency)
        # 创建假设索引
        info = []
        for i, index in enumerate(indexes):
            cur_cost = 0.0
            create_sql = index.create_hypopg_sql()
            cursor.execute(create_sql)
            logger.debug(f"创建假设索引: {index.name}")
            for q in queries:
                key = CostCache.hypo_key(q.sql, indexes[: i + 1])
                if (c := cost_cache.get(key)) is not None:
                    # logger.debug(f"查询 {q} 命中缓存")
                    cur_cost += c * q.frequency
                else:
                    cur_cost += cost_cache.update(key, explain_query(cursor, q.sql) * q.frequency)
            profit = info[-1].cost - cur_cost if info else base_cost - cur_cost
            info.append(IndexInfo(index=index, cost=cur_cost, profit=profit))
        return IdxSeqInfo(basic_cost=base_cost, infos=info)

    except Exception as e:
        logger.error(f"操作失败: {str(e)}")
        raise e
    finally:
        conn.rollback()
        if cursor:
            cursor.close()


def drop_hypo_index(cursor: Cursor, oid: int):
    assert isinstance(cursor, psycopg.Cursor), "hypopg 仅支持 PostgreSQL"
    cursor.execute(literal(f"SELECT hypopg_drop_index({oid});"))


def get_db_size(cursor: Cursor) -> float:
    """获取数据库大小, 单位为 Bytes."""
    if isinstance(cursor, psycopg.Cursor):
        cursor.execute("SELECT pg_database_size(current_database());")
        res = cursor.fetchone()
        assert res is not None, "get_db_size 未返回结果"
        return float(res[0])
    else:
        raise ValueError(f"Unsupported cursor type: {type(cursor)}")


def resolve_size_limits(
    db_option: DBOption,
    size_limit_mb: float | None,
    size_factor: float | None,
) -> dict[str, float | None]:
    """根据命令行配置计算各数据库的索引大小限制 (Bytes)."""
    if size_limit_mb is not None and size_limit_mb <= 0:
        raise ValueError("--size-limit 必须大于 0")
    if size_factor is not None and size_factor <= 0:
        raise ValueError("--size-factor 必须大于 0")
    if size_limit_mb is not None and size_factor is not None:
        raise ValueError("不能同时指定 --size-limit 与 --size-factor")

    limits: dict[str, float | None] = {}
    connections = db_option.connections
    if size_factor is not None:
        if db_option.db_type != "pg":
            raise ValueError("--size-factor 目前仅支持 PostgreSQL 数据库")
        seen_conn_size: dict[int, float] = {}
        for db_name, conn in connections.items():
            key = id(conn)
            if key not in seen_conn_size:
                cursor = conn.cursor()
                try:
                    seen_conn_size[key] = get_db_size(cursor)
                finally:
                    conn.rollback()
                    cursor.close()
            db_size = seen_conn_size[key]
            limit = db_size * size_factor
            limits[db_name] = limit
            logger.debug(
                "数据库 %s 体积 %.2f MB, 按因子 %.4f 计算索引上限为 %.2f MB",
                db_name,
                db_size / (1024 * 1024),
                size_factor,
                limit / (1024 * 1024),
            )
        return limits

    if size_limit_mb is None:
        return dict.fromkeys(connections, None)

    limit_bytes = float(size_limit_mb) * 1024 * 1024
    return dict.fromkeys(connections, limit_bytes)

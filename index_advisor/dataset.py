import json
import multiprocessing
from pathlib import Path
from typing import Mapping, Sequence

from .db import Connection, DBOption, explain_query, handle_sql, init_worker_connections, worker_connection
from .ia_logging import logger
from .schemas import DatabaseSchema, Index, parse_indexes_str, schemas
from .sql_workload_features import (
    QueryPredicateStats,
    aggregate_workload_stats,
    compute_sql_predicate_task,
    init_worker_state,
)
from .workload import QueryInfo, Workload


def read_file(file_path: str | Path) -> list[str]:
    """
    读取 workload 并获取 SQL 语句. 需保证每行一个 SQL 语句.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return [l.strip() for l in file.readlines() if l.strip()]


def load_queries_from_file(
    file_path: str | Path,
    opt: DBOption,
    explain: bool = True,
    execute: bool = False,
) -> set[QueryInfo]:
    """
    读取SQL文件并加载每个查询的相关信息.

    当任一查询出错时终止程序.

    Args:
        file_path: 输入文件路径
        opt: 数据库选项
        explain: 是否使用 EXPLAIN 获取查询开销
        execute: 是否执行查询以获取实际开销

    Returns:
        各查询的 QueryInfo
    """
    lines = read_file(file_path)
    logger.info(f"读取 {file_path}")
    queries = set()
    i = 0
    for line in lines:
        sql = line.strip()
        if not sql:
            continue
        i += 1
        q = process_query(sql, opt.schema, opt.conn, explain, execute)
        queries.add(q)
    logger.info(f"共处理 {i} 条 SQL, {len(queries)} 条有效")
    return queries


def process_query(sql: str, schema: DatabaseSchema, conn: Connection, explain: bool, execute: bool) -> QueryInfo:
    logger.debug(f"处理查询: {sql}")
    tables_columns, large_columns, indexes = handle_sql(sql, schema)
    logger.debug(f"获取行列: {tables_columns}")
    logger.debug(f"已有索引: {indexes}")
    try:
        cost = 0
        if explain:
            cost = explain_query(conn.cursor(), sql, False, True)
            logger.debug(f"基础开销: {cost:.2f}")
        cost_real = None
        if execute:
            cost_real = explain_query(conn.cursor(), sql, True, True)
            logger.debug(f"基础开销(执行): {cost_real:.2f}")
    except Exception as e:
        logger.error(f"处理 {sql}\n出错: {e}")
        exit(1)

    return QueryInfo(
        sql=sql,
        tables_columns=tables_columns,
        large_columns=large_columns,
        exist_indexes=indexes,
        basic_cost_explain=cost,
        basic_cost_real=cost_real,
    )


def load_index_eab_res(file_path: str | Path, schema: DatabaseSchema):
    """
    从 Index_eab 输出的 JSON 文件中读取工作负载及对应的索引推荐结果.

    Args:
        file_path: 输入 JSON 文件路径
        schema: 数据库模式信息, 用于验证索引有效性
    """

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    assert isinstance(data, list), "JSON 文件内容应为列表格式"

    logger.info(f"读取 JSON 文件 {file_path}")
    workloads: list[tuple[set[str], list[Index]]] = []

    for i, workload_res in enumerate(data):
        assert isinstance(workload_res, dict), "每个条目应为字典格式, 其 key 表示算法名称"
        assert len(workload_res) > 0, "应包含至少一个算法的结果"
        logger.debug(f"索引推荐算法: {workload_res.keys()} 将使用第一个算法的结果")

        algo_res = workload_res.values().__iter__().__next__()
        assert isinstance(algo_res, dict), "算法结果应为字典格式"

        # parse sqls
        assert (
            "workload" in algo_res
            and isinstance(algo_res["workload"], list)
            and len(algo_res["workload"]) > 0
            and all(isinstance(s, str) for s in algo_res["workload"])
        ), "算法结果中应包含 'workload' 键且其值为字符串列表"

        sqls: set[str] = {sql.strip() for sql in algo_res["workload"] if sql.strip()}

        # parse indexes
        assert (
            "indexes" in algo_res
            and isinstance(algo_res["indexes"], list)
            and all(isinstance(idx, str) for idx in algo_res["indexes"])
        ), "算法结果中应包含 'indexes' 键且其值为字符串列表"

        indexes = algo_res["indexes"]
        if len(indexes) == 0:
            logger.warning(f"工作负载 {sqls} 中没有索引推荐, 跳过")
            continue
        # 不剔除启发式算法推荐的已存在索引
        advised_indexes, (invalid, exist, duplicate) = parse_indexes_str("\n".join(indexes), schema, True)
        logger.info(f"推荐索引: {advised_indexes}")
        logger.info(f"过滤索引: 无效 {invalid} 已存在 {exist} 重复 {duplicate}")
        if len(advised_indexes) == 0:
            logger.warning(f"工作负载 {sqls} 中没有有效索引推荐, 跳过")
            continue

        workloads.append((sqls, advised_indexes))
    return workloads


def rebuild(w: Workload, opt: DBOption, explain: bool, execute: bool):
    """
    保持 db, id, sqls 不变, 重新构造 QueryInfo 和 Workload.

    根据 explain 和 execute 决定是否重新获取 basic_cost_explain 和 basic_cost_real.
    """
    queries = set()
    for raw_q in w.queries:
        q = process_query(raw_q.sql, schemas[w.db], opt.connections[w.db], explain, execute)
        if not explain:
            q.basic_cost_explain = raw_q.basic_cost_explain
        if not execute:
            q.basic_cost_real = raw_q.basic_cost_real
        queries.add(q)

    return Workload(w.db, w.id, queries, labels=w.labels, label_overrides=w.label_overrides)


def init_workload_predicate_stats(
    workloads: Sequence[Workload],
    opt: DBOption,
    schema_names: Mapping[str, str] | str = "public",
    workers: int | None = None,
):
    """预计算并写回 workloads 的 predicate_stats."""
    workload_list = list(workloads)
    if not workload_list:
        return []

    worker_count = 1 if workers is None else max(1, workers)
    required_dbs = sorted({w.db for w in workload_list})
    db_schemas = {db: schemas[db] for db in required_dbs if db in schemas}
    opt_dict = opt.to_dict() | {"databases": required_dbs}

    sqls_by_db: dict[str, set[str]] = {}
    for workload in workload_list:
        sqls_by_db.setdefault(workload.db, set()).update(q.sql for q in workload.queries)

    if not sqls_by_db:
        return []

    per_query_stats_by_db: dict[str, dict[str, QueryPredicateStats]] = {}

    tasks = [(db_name, sql) for db_name, sqls in sqls_by_db.items() for sql in sorted(sqls)]
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(
        processes=worker_count,
        initializer=init_worker_state,
        initargs=(opt_dict, db_schemas, schema_names),
    ) as pool:
        try:
            results = pool.map(compute_sql_predicate_task, tasks)
        except Exception:
            logger.exception("failed to compute predicate stats")
            raise

        for db_name, sql, stats in results:
            per_query_stats_by_db.setdefault(db_name, {})[sql] = stats

    for workload in workload_list:
        per_query_stats = per_query_stats_by_db.get(workload.db, {})
        stats = aggregate_workload_stats(workload, per_query_stats)
        workload.predicate_stats = stats


def init_schema_column_stats(opt: DBOption, workers: int | None = None):
    """为指定数据库预计算所有列的 row_count 与 distinct_count 缓存."""

    worker_count = 1 if workers is None else max(1, workers)
    target_dbs = sorted({db for db in opt.databases if db in schemas})

    if worker_count == 1:
        for db in target_dbs:
            schema = schemas[db]
            conn = opt.connect(db)
            try:
                for table_name, table_schema in schema.tables.items():
                    row_count = schema.column_statistic(conn, table_name, table_schema.any_col, "row_count")
                    for col in table_schema.column_names:
                        schema.column_stats[(table_name, col, "row_count")] = row_count
                        schema.column_stats[(table_name, col, "distinct_count")] = schema.column_statistic(
                            conn, table_name, col, "distinct_count"
                        )
            finally:
                conn.close()
        return

    tasks = []
    for db in target_dbs:
        schema = schemas[db]
        tasks.extend((db, table_name) for table_name in schema.tables)

    ctx = multiprocessing.get_context("fork")
    opt_dict = opt.to_dict() | {"databases": target_dbs}
    with ctx.Pool(processes=worker_count, initializer=init_worker_connections, initargs=(opt_dict,)) as pool:
        for db, table_name, row_count, distinct_counts in pool.imap_unordered(_collect_table_stats_task, tasks):
            schema = schemas.get(db)
            if schema is None:
                continue
            table_schema = schema.tables[table_name]
            for col in table_schema.column_names:
                schema.column_stats[(table_name, col, "row_count")] = row_count
                schema.column_stats[(table_name, col, "distinct_count")] = distinct_counts.get(col)


def _collect_table_stats_task(
    task: tuple[str, str],
) -> tuple[str, str, int | float | None, dict[str, int | float | None]]:
    db, table_name = task
    schema = schemas.get(db)
    if schema is None:
        raise ValueError(f"Missing schema definition for database {db}")
    conn = worker_connection(db)
    table_schema = schema.tables[table_name]
    row_count = schema.column_statistic(conn, table_name, table_schema.any_col, "row_count")
    logger.info(f"预计算 {db}.{table_name} 行数: {row_count}")
    distinct_counts = {
        col: schema.column_statistic(conn, table_name, col, "distinct_count") for col in table_schema.column_names
    }
    logger.info(f"预计算 {db}.{table_name} distinct 行数: {distinct_counts}")
    return db, table_name, row_count, distinct_counts

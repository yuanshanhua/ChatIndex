import argparse
import multiprocessing
import statistics
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence

from index_advisor.db import DBOption, explain_query, get_hypo_index_size, init_worker_connections, worker_connection
from index_advisor.eval import ModelResult, Result, load_eval_results
from index_advisor.ia_logging import log_to_file, logger
from index_advisor.schemas import DatabaseSchema, Index, parse_indexes_str, schemas
from index_advisor.size_limit_utils import DBLimitPlan, SizeScenario, build_size_scenarios, collect_db_limit_plan


@dataclass
class PlannedIndex:
    text: str
    definition: Index
    estimated_size_bytes: float


@dataclass
class IndexExecutionDetail:
    text: str
    estimated_size_bytes: float
    actual_size_bytes: float


@dataclass
class WorkloadLimitMetrics:
    limit_bytes: float | None
    total_estimated_size_bytes: float
    total_actual_size_bytes: float
    query_times_ms: List[float]
    index_details: List[IndexExecutionDetail]


@dataclass
class ExecutionRecord:
    checkpoint_name: str
    checkpoint_id: int
    workload_id: int
    db: str
    limit_bytes: float | None
    limit_label: str
    total_actual_size_bytes: float
    total_estimated_size_bytes: float
    total_time_ms: float
    query_times_ms: List[float]
    index_details: List[IndexExecutionDetail]


LOG = logger.getChild("execute")


def calculate_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    if len(values) == 1:
        return {"max": values[0], "min": values[0], "mean": values[0], "median": values[0], "std": 0.0}
    return {
        "max": max(values),
        "min": min(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values),
    }


def bytes_to_mb(value: float) -> float:
    return value / (1024 * 1024)


def format_limit_label(limit_bytes: float | None) -> str:
    if limit_bytes is None:
        return "unlimited"
    return f"{bytes_to_mb(limit_bytes):.3f} MB"


def filter_model_results_by_checkpoints(
    model_results: List[ModelResult], checkpoint_ids: Sequence[int]
) -> List[ModelResult]:
    targets = set(checkpoint_ids)
    filtered = [mr for mr in model_results if mr.id in targets]
    missing = targets - {mr.id for mr in filtered}
    if missing:
        LOG.warning(f"警告: 未找到 checkpoint id: {', '.join(str(i) for i in sorted(missing))}")
    return filtered


def filter_model_results_by_db(model_results: List[ModelResult], databases: List[str]) -> int:
    if not databases:
        return 0
    targets = {db.lower() for db in databases}
    removed = 0
    for model_result in model_results:
        original_count = len(model_result.results)
        model_result.results = [r for r in model_result.results if (r.db or "").lower() in targets]
        removed += original_count - len(model_result.results)
    return removed


def filter_model_results_by_workloads(model_results: List[ModelResult], workload_ids: Sequence[int] | None) -> int:
    if not workload_ids:
        return 0
    targets = set(workload_ids)
    removed = 0
    for model_result in model_results:
        original = len(model_result.results)
        model_result.results = [r for r in model_result.results if r.workload_id in targets]
        removed += original - len(model_result.results)
    return removed


def ensure_databases_covered(model_results: List[ModelResult], args: argparse.Namespace):
    all_db = {r.db for mr in model_results for r in mr.results if r.db}
    all_db = sorted(all_db)
    if args.db:
        all_db.extend(args.db)
    args.db = list(dict.fromkeys(all_db).keys())
    if all_db:
        LOG.info(f"自动添加数据库: {', '.join(sorted(all_db))}")


def prepare_index_plan(
    result: Result,
    conn,
    schema: DatabaseSchema,
    db_type: str,
) -> List[PlannedIndex]:
    if result.db is None:
        raise ValueError("workload 缺少 db 字段, 无法构建 index plan")
    db_name = result.db
    sources: List[tuple[str, float | None]] = []
    if result.indexes:
        sources = [(idx.index, idx.size) for idx in result.indexes]
    elif result.response:
        sources = [(line.strip(), None) for line in result.response.splitlines() if line.strip()]

    if not sources:
        return []

    cursor = conn.cursor()
    plan: List[PlannedIndex] = []
    try:
        for text, size_bytes in sources:
            indexes, _ = parse_indexes_str(text, schema)
            if not indexes:
                continue
            idx = indexes[0]
            estimated_size = size_bytes
            if estimated_size is None:
                if db_type != "pg":
                    raise ValueError(
                        f"索引 {text} 缺少大小信息且当前数据库类型 {db_type} 不支持自动估算，请在评估结果中包含 size 字段"
                    )
                estimated_size = get_hypo_index_size(cursor, db_name, idx)
            plan.append(PlannedIndex(text=text, definition=idx, estimated_size_bytes=float(estimated_size)))
    finally:
        conn.rollback()
        cursor.close()
    return plan


def fetch_index_actual_size(cursor, index_name: str) -> float:
    cursor.execute("SELECT pg_relation_size(%s::regclass);", (index_name,))
    res = cursor.fetchone()
    if not res:
        raise ValueError(f"pg_relation_size 未返回结果: {index_name}")
    return float(res[0])


def measure_query_time(cursor, query: str, runs: int) -> float:
    durations: List[float] = []
    # normalized_sql = " ".join(query.strip().split())
    # truncated_sql = normalized_sql[:120] + ("..." if len(normalized_sql) > 120 else "")
    for _ in range(runs):
        durations.append(explain_query(cursor, query, analyze=True, raise_exception=True))
    avg_time = sum(durations) / len(durations)
    # LOG.debug(f"Query 平均耗时 {avg_time:.2f} ms, runs={runs}, sql={truncated_sql}")
    return avg_time


def _measure_query_time_worker(task: tuple[str, int, str]) -> float:
    db_name, runs, query = task
    conn = worker_connection(db_name)
    cursor = conn.cursor()
    try:
        return measure_query_time(cursor, query, runs)
    finally:
        conn.rollback()
        cursor.close()


def execute_workload_with_limits(
    db_option: DBOption,
    db_name: str,
    queries: Sequence[str],
    plan: List[PlannedIndex],
    limits: Sequence[float | None],
    query_runs: int,
    worker: int,
) -> Dict[float | None, WorkloadLimitMetrics]:
    conn = db_option.connections[db_name]
    cursor = conn.cursor()
    metrics: Dict[float | None, WorkloadLimitMetrics] = {}
    created = 0
    total_estimated = 0.0
    total_actual = 0.0
    index_details: List[IndexExecutionDetail] = []
    pool = None
    if worker > 1:
        ctx = multiprocessing.get_context("fork")
        opt_dict = db_option.to_dict() | {"databases": [db_name]}
        pool = ctx.Pool(processes=worker, initializer=init_worker_connections, initargs=(opt_dict,))
    try:
        for limit in limits:
            target = float("inf") if limit is None else float(limit)
            while created < len(plan):
                next_index = plan[created]
                if limit is not None and total_estimated + next_index.estimated_size_bytes > target + 1e-9:
                    break
                cursor.execute(next_index.definition.create_sql())
                actual_size = fetch_index_actual_size(cursor, next_index.definition.name)
                total_estimated += next_index.estimated_size_bytes
                total_actual += actual_size
                index_details.append(
                    IndexExecutionDetail(
                        text=next_index.text,
                        estimated_size_bytes=next_index.estimated_size_bytes,
                        actual_size_bytes=actual_size,
                    )
                )
                created += 1
            else:
                LOG.warning(
                    f"所有推荐索引均已创建, 无法再应用更高的 limit. 当前索引总大小(估计): {bytes_to_mb(total_estimated):.3f} MB, 实际: {bytes_to_mb(total_actual):.3f} MB"
                )
            LOG.info(f"应用 limit: {format_limit_label(limit)}, 已创建索引: {created}, 剩余: {len(plan) - created}")
            if worker == 1:
                query_times = [measure_query_time(cursor, sql, query_runs) for sql in queries]
            else:
                assert pool is not None
                conn.commit()
                query_times = pool.map(
                    _measure_query_time_worker, [(db_name, query_runs, sql) for sql in queries], chunksize=1
                )
            metrics[limit] = WorkloadLimitMetrics(
                limit_bytes=limit,
                total_estimated_size_bytes=total_estimated,
                total_actual_size_bytes=total_actual,
                query_times_ms=query_times,
                index_details=list(index_details),
            )
        return metrics
    finally:
        try:
            if worker == 1:
                conn.rollback()
            else:
                conn.rollback()
                if created > 0:
                    drop_cursor = conn.cursor()
                    try:
                        for idx in plan[:created]:
                            drop_cursor.execute(idx.definition.drop_sql())
                        conn.commit()
                    finally:
                        drop_cursor.close()
        finally:
            cursor.close()
            if pool is not None:
                pool.close()
                pool.join()


def collect_execution_records(
    model_results: List[ModelResult],
    db_option: DBOption,
    scenarios: List[SizeScenario],
    db_limit_plan: Dict[str, DBLimitPlan],
    query_runs: int,
    worker: int,
) -> Dict[str, List[ExecutionRecord]]:
    scenario_records: Dict[str, List[ExecutionRecord]] = {scenario.key: [] for scenario in scenarios}
    connections = db_option.connections

    for model_result in model_results:
        LOG.info(f"开始评估 {model_result.name} (ID: {model_result.id})")
        for result in model_result.results:
            db_name = result.db
            if not db_name:
                LOG.warning(f"跳过 workload {result.workload_id}: 未提供数据库名称")
                continue
            if db_name not in connections:
                LOG.warning(f"跳过 workload {result.workload_id}: 未找到数据库连接 {db_name}")
                continue
            if not result.queries:
                LOG.warning(f"跳过 workload {result.workload_id}: 没有查询语句")
                continue
            schema = schemas.get(db_name)
            if schema is None:
                LOG.warning(f"跳过 workload {result.workload_id}: 未知 schema {db_name}")
                continue

            plan = prepare_index_plan(result, connections[db_name], schema, db_option.db_type)
            LOG.info(f"workload {result.workload_id} ({db_name}) 总推荐索引数: {len(plan)}")
            limit_plan = db_limit_plan.get(db_name, DBLimitPlan(tuple(), True))
            limit_sequence: List[float | None] = list(limit_plan.finite_limits)
            if limit_plan.include_unbounded or not limit_sequence:
                limit_sequence.append(None)
            LOG.info(f"workload {result.workload_id} ({db_name}) limit 方案数量: {len(limit_sequence)}")

            metrics_by_limit = execute_workload_with_limits(
                db_option, db_name, result.queries, plan, limit_sequence, query_runs, worker
            )

            for scenario in scenarios:
                limit_bytes = scenario.limits.get(db_name)
                key = limit_bytes if limit_bytes is not None else None
                metrics = metrics_by_limit.get(key)
                if metrics is None:
                    continue
                total_time = sum(metrics.query_times_ms)
                scenario_records[scenario.key].append(
                    ExecutionRecord(
                        checkpoint_name=model_result.name,
                        checkpoint_id=model_result.id,
                        workload_id=result.workload_id,
                        db=db_name,
                        limit_bytes=limit_bytes,
                        limit_label=format_limit_label(limit_bytes),
                        total_actual_size_bytes=metrics.total_actual_size_bytes,
                        total_estimated_size_bytes=metrics.total_estimated_size_bytes,
                        total_time_ms=total_time,
                        query_times_ms=metrics.query_times_ms,
                        index_details=metrics.index_details,
                    )
                )
            total_records = sum(len(v) for v in scenario_records.values())
            LOG.info(f"workload {result.workload_id} ({db_name}) 执行完成, 累计记录 {total_records} 条")
    return scenario_records


def print_execution_summary(scenario: SizeScenario, records: List[ExecutionRecord]):
    LOG.info("=" * 80)
    LOG.info(f"场景评估: {scenario.label}")
    LOG.info("=" * 80)
    if not records:
        LOG.info("没有匹配的执行记录")
        return

    by_checkpoint: Dict[int, List[ExecutionRecord]] = {}
    for record in records:
        by_checkpoint.setdefault(record.checkpoint_id, []).append(record)

    for checkpoint_id in sorted(by_checkpoint):
        checkpoint_records = by_checkpoint[checkpoint_id]
        checkpoint_name = checkpoint_records[0].checkpoint_name
        LOG.info(f"Checkpoint {checkpoint_name} (ID: {checkpoint_id})")
        times = [r.total_time_ms for r in checkpoint_records]
        sizes_mb = [bytes_to_mb(r.total_actual_size_bytes) for r in checkpoint_records]
        time_stats = calculate_stats(times)
        size_stats = calculate_stats(sizes_mb)
        if time_stats:
            LOG.info(
                f"  查询耗时统计 (ms): max={time_stats['max']:.2f} min={time_stats['min']:.2f} "
                f"mean={time_stats['mean']:.2f} median={time_stats['median']:.2f} std={time_stats['std']:.2f}"
            )
        if size_stats:
            LOG.info(
                f"  索引体积统计 (MB): max={size_stats['max']:.3f} min={size_stats['min']:.3f} "
                f"mean={size_stats['mean']:.3f} median={size_stats['median']:.3f} std={size_stats['std']:.3f}"
            )
        LOG.info(f"  workload 数量: {len(checkpoint_records)}")

        for record in sorted(checkpoint_records, key=lambda r: r.workload_id):
            LOG.info(
                f"    workload {record.workload_id} ({record.db}) limit={record.limit_label}: "
                f"time={record.total_time_ms:.2f} ms, indexes={len(record.index_details)}, "
                f"est_size={bytes_to_mb(record.total_estimated_size_bytes):.3f} MB "
                f"actual_size={bytes_to_mb(record.total_actual_size_bytes):.3f} MB"
            )
            for detail in record.index_details:
                LOG.info(
                    f"      - {detail.text}: est={bytes_to_mb(detail.estimated_size_bytes):.3f} MB, "
                    f"actual={bytes_to_mb(detail.actual_size_bytes):.3f} MB"
                )


def main():
    parser = argparse.ArgumentParser(description="对 checkpoint 推荐结果创建真实索引并测量执行时间")
    DBOption.add_arg_group_to(parser)
    parser.add_argument("file_path", help="评估结果 JSON 文件路径")
    parser.add_argument("--worker", "-w", type=int, default=1, help="并发执行查询的进程数. 当大于 1 时将不使用长事务.")
    parser.add_argument(
        "--checkpoint-id", "--cid", type=int, nargs="+", required=True, help="需要评估的 checkpoint id"
    )
    parser.add_argument("--workload-id", "--wid", type=int, nargs="+", help="仅评估指定 workload id", default=None)
    parser.add_argument(
        "--size-limit", "-s", type=float, nargs="+", help="索引大小上限 (MB), 可指定多个值", default=None
    )
    parser.add_argument(
        "--size-factor", type=float, nargs="+", help="按数据库体积比例设置索引上限, 可指定多个值", default=None
    )
    parser.add_argument("--filter-db", "-d", nargs="+", help="仅评估指定数据库", default=None)
    parser.add_argument("--query-runs", type=int, default=3, help="每条查询执行次数, 计算平均耗时")
    parser.add_argument("--debug", action="store_true", help="启用调试日志")
    args = parser.parse_args()

    if args.dbtype != "pg":
        raise ValueError("此脚本目前仅支持 PostgreSQL 数据库类型")
    if args.query_runs <= 0:
        raise ValueError("--query-runs 必须大于 0")

    log_to_file("execute", "INFO" if not args.debug else "DEBUG")

    checkpoint_labels = ", ".join(str(cid) for cid in args.checkpoint_id)
    LOG.info(f"开始执行评估: file={args.file_path}, checkpoint={checkpoint_labels}, query_runs={args.query_runs}")

    model_results = load_eval_results(args.file_path)
    model_results = filter_model_results_by_checkpoints(model_results, args.checkpoint_id)
    if not model_results:
        LOG.info("没有匹配的 checkpoint")
        return

    if args.filter_db:
        removed = filter_model_results_by_db(model_results, args.filter_db)
        LOG.info(f"数据库筛选移除 {removed} 个 workload")

    removed_workloads = filter_model_results_by_workloads(model_results, args.workload_id)
    if removed_workloads:
        LOG.info(f"workload 筛选移除 {removed_workloads} 个记录")

    model_results = [mr for mr in model_results if mr.results]
    if not model_results:
        LOG.info("筛选后没有可评估的 workload")
        return

    ensure_databases_covered(model_results, args)

    db_option = DBOption.from_args(args)
    scenarios = build_size_scenarios(db_option, args.size_limit, args.size_factor)
    size_labels = ", ".join(f"{scenario.label}" for scenario in scenarios) or "无"
    LOG.info(f"构建 size 场景: {size_labels}")
    db_limit_plan = collect_db_limit_plan(scenarios)

    scenario_records = collect_execution_records(
        model_results, db_option, scenarios, db_limit_plan, args.query_runs, args.worker
    )

    any_record = any(scenario_records.values())
    if not any_record:
        LOG.info("未生成任何执行记录")
        return

    for scenario in scenarios:
        records = scenario_records.get(scenario.key, [])
        print_execution_summary(scenario, records)

    LOG.info("执行评估完成")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        LOG.exception(f"执行失败: {exc}")
        sys.exit(1)

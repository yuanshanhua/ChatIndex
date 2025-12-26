import multiprocessing
from typing import Any

from .db import Connection, CostCache, DBOption, SizeCache, create_hypo_index_and_get_all_info, get_hypo_index_size
from .workload import Workload


def _populate_caches(
    workloads: list[Workload],
    connections: dict[str, Connection],
    cost_cache: CostCache,
    size_cache: SizeCache,
) -> None:
    for workload in workloads:
        conn = connections[workload.db]
        labels = workload.labels or []
        create_hypo_index_and_get_all_info(conn, labels, workload.queries, cost_cache)
        cursor = conn.cursor()
        try:
            for idx in labels:
                get_hypo_index_size(cursor, workload.db, idx, size_cache)
        finally:
            conn.rollback()
            cursor.close()


def _build_cache_worker(args: tuple[list[Workload], dict[str, Any]]) -> tuple[dict, dict]:
    workloads, db_option_dict = args
    db_option = DBOption.from_dict(db_option_dict)
    cost_cache = CostCache()
    size_cache = SizeCache()
    connections = db_option.connections
    try:
        _populate_caches(workloads, connections, cost_cache, size_cache)
        return cost_cache._cache, size_cache._cache
    finally:
        closed: set[int] = set()
        for conn in connections.values():
            conn_id = id(conn)
            if conn_id in closed:
                continue
            closed.add(conn_id)
            try:
                conn.close()
            except Exception:
                pass


def build_cache(
    workloads: list[Workload],
    db_option: DBOption,
    workers: int,
    cost_cache: CostCache | None = None,
    size_cache: SizeCache | None = None,
) -> tuple[CostCache, SizeCache]:
    if cost_cache is None:
        cost_cache = CostCache()
    if size_cache is None:
        size_cache = SizeCache()

    if not workloads:
        return cost_cache, size_cache

    if workers <= 1:
        _populate_caches(workloads, db_option.connections, cost_cache, size_cache)
        return cost_cache, size_cache

    # 工作负载过少时, 将每个工作负载拆分为单查询, 以提高并行度
    if len(workloads) < workers * 2:
        splited: list[Workload] = []
        for w in workloads:
            splited.extend(w.to_1q())
        workloads = splited

    worker_count = max(1, min(workers, len(workloads)))
    chunk_size = (len(workloads) + worker_count - 1) // worker_count
    tasks: list[tuple[list[Workload], dict[str, Any]]] = []
    db_option_dict = db_option.to_dict()
    for i in range(0, len(workloads), chunk_size):
        tasks.append((workloads[i : i + chunk_size], db_option_dict))

    ctx = multiprocessing.get_context("spawn")
    results: list[tuple[dict, dict]] = []
    with ctx.Pool(processes=worker_count) as pool:
        results = pool.map(_build_cache_worker, tasks)

    for cache_dict, size_dict in results:
        cost_cache._cache.update(cache_dict)
        size_cache._cache.update(size_dict)
    return cost_cache, size_cache

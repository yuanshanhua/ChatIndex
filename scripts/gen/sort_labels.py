import argparse
import atexit
import logging
import multiprocessing as mp
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from index_advisor.db import (
    Connection,
    DBOption,
    create_hypo_index_and_get_cost,
    drop_hypo_index,
    explain_query,
    get_hypo_index_size,
)
from index_advisor.ia_logging import log_to_file, logger
from index_advisor.schemas.schema import Index
from index_advisor.workload import QueryInfo, Workload, load_workloads, resolve_databases, save_workloads


logger = logger.getChild("sort_labels")


_worker_db_option: DBOption | None = None
_worker_best = False


def _cleanup_worker_connections() -> None:
    global _worker_db_option
    if _worker_db_option is None:
        return
    connections = _worker_db_option.__dict__.get("connections")
    if not connections:
        _worker_db_option = None
        return
    for conn in set(connections.values()):
        try:
            conn.close()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to close worker connection: %s", exc)
    _worker_db_option = None


def _init_worker(db_option_dict: dict, best: bool, debug: bool) -> None:
    global _worker_db_option, _worker_best
    _worker_db_option = DBOption.from_dict(db_option_dict)
    _worker_best = best
    parent_logger = logger.parent
    if parent_logger:
        level = logging.DEBUG if debug else logging.INFO
        for handler in parent_logger.handlers:
            handler.setLevel(level)
    atexit.register(_cleanup_worker_connections)


def _process_workload(workload: Workload) -> tuple[bool, Workload]:
    if _worker_db_option is None:
        raise RuntimeError("Worker database options not initialized")
    changed = sort_labels(workload, _worker_db_option, _worker_best)
    return changed, workload


def create_hypo_index(cursor, idx: Index) -> tuple[int, float]:
    """Create a single hypopg index and return its oid and size(MB)."""
    cursor.execute(idx.create_hypopg_sql())
    res = cursor.fetchone()
    if not res:
        raise RuntimeError(f"hypopg_create_index 未返回结果: {idx}")
    oid_info = res[0]
    oid = oid_info[0] if isinstance(oid_info, (list, tuple)) else oid_info
    cursor.execute("SELECT * FROM hypopg_relation_size(%s);", (oid,))
    size_row = cursor.fetchone()
    size = float(size_row[0]) / (1024 * 1024) if size_row and size_row[0] else 0.0
    return int(oid), size


def get_cost(cursor, queries: Iterable[QueryInfo]) -> float:
    cost = 0.0
    try:
        for q in queries:
            cost += explain_query(cursor, q.sql) * q.frequency
        return cost
    except Exception as exc:
        logger.error("Failed to evaluate hypothetical indexes: %s", exc)
        return -1


def search_best_order(conn: Connection, w: Workload) -> list[Index]:
    if not w.labels:
        return []
    labels = set(w.labels)
    candidates = set(labels)
    cursor = conn.cursor()
    best_order: list[Index] = []
    try:
        cursor.execute("SELECT hypopg_reset();")
        base_cost = get_cost(cursor, w.queries)
        no_profit = 0
        for _ in labels:
            best_idx = None
            best_score = float("-inf")
            best_cost = float("inf")
            # 寻找当前位置最优索引
            for idx in candidates:
                oid, size = create_hypo_index(cursor, idx)
                candidate_cost = get_cost(cursor, w.queries)
                drop_hypo_index(cursor, oid)
                score = 0.0
                if candidate_cost > 0 and size > 0:
                    score = (base_cost - candidate_cost) / base_cost / size
                logger.debug(f"Workload {w.id} evaluating index: {idx}, score: {score:.6f}")
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_cost = candidate_cost
            # 候选均无收益, 提前终止
            if best_score <= 1e-9:
                no_profit += 1
            if no_profit >= 3:
                logger.debug(
                    f"Workload {w.id}(hash: {w.sqls_hash}) 连续 {no_profit} 轮无收益提前终止, 已选 {len(best_order)}"
                )
                break
            # 创建并添加最优索引
            assert best_idx is not None, "No best index found"
            cursor.execute(best_idx.create_hypopg_sql())
            base_cost = best_cost
            logger.debug(f"Workload {w.id}(hash: {w.sqls_hash}) 选中索引 {best_idx}, score={best_score:.6f}")
            best_order.append(best_idx)
            candidates.remove(best_idx)
        return best_order
    finally:
        try:
            cursor.execute("SELECT hypopg_reset();")
        finally:
            cursor.close()
            conn.rollback()


def get_index_scores(
    conn: Connection,
    w: Workload,
    exists: Iterable[Index],
    candidates: Iterable[Index],
) -> dict[Index, float]:
    """
    给定一系列查询和已有索引集, 计算待定索引集中各索引的单位体积收益.

    Args:
        q (set[QueryInfo]): 查询集
        exists (Iterable[Index]): 已存在索引集
        candidates (Iterable[Index]): 待定索引集

    Returns:
        dict: 每个待定索引的单位体积收益
    """
    exists_ = set(exists)
    candidates_ = set(candidates) - exists_
    basic_cost = create_hypo_index_and_get_cost(conn, exists, w.queries)
    scores = {}
    for idx in candidates_:
        cost_with_index = create_hypo_index_and_get_cost(conn, exists_ | {idx}, w.queries)
        cost_improvement = basic_cost - cost_with_index
        with conn.cursor() as cur:
            size = get_hypo_index_size(cur, w.db, idx) / (1024 * 1024)  # MB
        scores[idx] = cost_improvement / basic_cost / size if size > 0 else 0
    return scores


def sort_labels(w: Workload, db_option: DBOption, best: bool) -> bool:
    if not w.labels:
        logger.info(f"Skipping workload {w} with no labels")
        return False

    conn = db_option.connections[w.db]
    logger.debug(f"Workload {w.id} 原始标签({len(w.labels)}):\n{w.sprint_labels}")

    if not best:
        # 仅分别考虑每个 label 的收益, 复杂度 O(n)
        scores = get_index_scores(conn, w, [], w.labels)
        raw_order = "\n".join(str(idx) for idx in w.labels)
        w.labels.sort(key=lambda idx: scores.get(idx, 0), reverse=True)
        w.label_overrides.clear()
        new_order = "\n".join(str(idx) for idx in w.labels)
        if raw_order != new_order:
            logger.info(f"Workload {w.id}(hash: {w.sqls_hash}) sorted labels:\n{new_order}")
        return raw_order != new_order

    # 逐个确认最好的索引, 复杂度 O(n^2)
    best_order = search_best_order(conn, w)
    changed = w.labels != best_order
    w.labels = best_order
    w.label_overrides.clear()
    if changed:
        logger.info(f"Workload {w.id}(hash: {w.sqls_hash}) 标签已更新")
        logger.debug(f"Workload {w.id}(hash: {w.sqls_hash}) 新标签:\n{w.sprint_labels}")
    return changed


def main():
    parser = argparse.ArgumentParser(
        description="按 cost/size 对 workload 的 labels 进行排序",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("workload_file", help="输入 workload 文件路径")
    parser.add_argument(
        "-o",
        "--output",
        help="输出 workload 文件路径. 若未指定则在原文件名后附加时间戳",
    )
    parser.add_argument("--simple", action="store_true", help="简易排序模式, 仅考虑单个标签收益")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--worker", "-w", type=int, default=32, help="并发数")
    DBOption.add_arg_group_to(parser)

    args = parser.parse_args()

    workload_path = Path(args.workload_file)
    workloads = load_workloads(workload_path)
    dbs = resolve_databases(workloads)
    args.db = list(dbs)

    db_option = DBOption.from_args(args)
    log_to_file("sort_labels", "DEBUG" if args.debug else "INFO")

    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%y%m%d%H%M%S")
        output_path = workload_path.with_name(f"{workload_path.stem}.sorted-{ts}{workload_path.suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    changed = 0
    worker_count = max(1, args.worker)
    if worker_count > 1 and len(workloads) > 1:
        worker_count = min(worker_count, len(workloads))
        logger.info("Using %s worker processes for %s workloads", worker_count, len(workloads))
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=worker_count,
            initializer=_init_worker,
            initargs=(db_option.to_dict(), not args.simple, args.debug),
        ) as pool:
            processed_workloads: list[Workload] = []
            for changed_flag, workload in pool.imap(_process_workload, workloads):
                changed += 1 if changed_flag else 0
                processed_workloads.append(workload)
        workloads = processed_workloads
    else:
        for w in workloads:
            changed += sort_labels(w, db_option, not args.simple)

    cached_connections = db_option.__dict__.get("connections")
    if cached_connections:
        for conn in set(cached_connections.values()):
            try:
                conn.close()
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.warning("Failed to close connection: %s", exc)

    save_workloads(output_path, workloads)
    logger.info(f"更新完成: {changed}/{len(workloads)} 个 workload 的标签发生变化")
    logger.info(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()

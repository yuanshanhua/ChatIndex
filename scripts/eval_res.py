import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import time
from typing import List, Tuple

from index_advisor import compute_reward
from index_advisor.cache import build_cache
from index_advisor.db import (
    CostCache,
    DBOption,
    SizeCache,
    create_hypo_index_and_get_all_info,
    create_index_and_get_cost,
    get_hypo_index_size,
)
from index_advisor.eval import IndexRes, ModelResult, Result, load_eval_results
from index_advisor.ia_logging import log_to_file, logger
from index_advisor.schemas import schemas
from index_advisor.schemas.schema import Index, parse_indexes_str
from index_advisor.workload import QueryInfo, Workload


LOG = logger.getChild("eval_res")

cost_cache = CostCache()
size_cache = SizeCache()


def _build_workload(res: Result, conn) -> tuple[Workload, List[Index], Tuple[int, int, int]]:
    if not res.db:
        raise ValueError("结果缺少数据库名称")
    if not res.response:
        LOG.warning(f"工作负载 {res.workload_id} 没有索引响应")
    schema = schemas[res.db]
    indexes, counts = parse_indexes_str(res.response, schema, keep_exist=True)

    cursor = conn.cursor()
    queries = set()
    try:
        for sql in res.queries:
            queries.add(
                QueryInfo(
                    sql=sql,
                    tables_columns={},
                    large_columns={},
                    exist_indexes=set(),
                    basic_cost_explain=-1,
                    basic_cost_real=None,
                    frequency=1,
                )
            )
    finally:
        conn.rollback()
        cursor.close()

    workload = Workload(db=res.db, id=res.workload_id, queries=queries, labels=indexes)
    return workload, indexes, counts


def _evaluate_workload(
    workload: Workload, indexes: List[Index], counts: Tuple[int, int, int], conn, real: bool
) -> Result:
    invalid, exist, duplicate = counts
    cursor = conn.cursor()
    try:
        idx_seq_info = create_hypo_index_and_get_all_info(conn, indexes, workload.queries, cost_cache)
        workload.basic_cost_explain = idx_seq_info.basic_cost
        actual_cost_hypo = idx_seq_info.indexed_cost

        index_res: List[IndexRes] = []
        res_indexes = []
        total_size = 0.0
        for info in idx_seq_info.infos:
            idx = info.index
            size = get_hypo_index_size(cursor, workload.db, idx, size_cache)
            total_size += size
            res_indexes.append(idx)
            index_res.append(
                IndexRes(index=str(idx), cost_hypo=info.cost, size=size, total_size=total_size, profit=info.profit)
            )
        actual_cost_real = None
        if real:
            actual_cost_real = create_index_and_get_cost(conn, res_indexes, workload.queries, True, cost_cache)
    finally:
        conn.rollback()
        cursor.close()

    total_size_mb = total_size / (1024 * 1024) if total_size else 0.0
    reward_raw = compute_reward(
        workload.basic_cost,
        actual_cost_real if actual_cost_real is not None else actual_cost_hypo,
        total=1,
        invalid=invalid,
        exist=exist,
        duplicate=duplicate,
    )
    reward = reward_raw / total_size_mb if total_size_mb > 0 else 0.0

    return Result(
        workload_id=workload.id,
        db=workload.db,
        queries=list(workload.sqls),
        sqls_hash=workload.sqls_hash,
        indexes=index_res,
        basic_cost_explain=workload.basic_cost_explain,
        basic_cost_real=None,
        actual_cost_hypo=actual_cost_hypo,
        actual_cost_real=actual_cost_real,
        reward=reward,
        prompt="",
        response="\n".join(str(idx) for idx in res_indexes),
        invalid=invalid,
        exist=exist,
        duplicate=duplicate,
        useless=0,
    )


def evaluate_model_result(model_result: ModelResult, db_option: DBOption, workers: int, real: bool) -> ModelResult:
    workloads = []
    meta: list[tuple[List[Index], Tuple[int, int, int]]] = []
    for res in model_result.results:
        assert res.db is not None
        conn = db_option.connections[res.db]
        workload, indexes, counts = _build_workload(res, conn)
        workloads.append(workload)
        meta.append((indexes, counts))
    LOG.info("构建成本和大小缓存...")
    st = time()
    build_cache(workloads, db_option, workers, cost_cache, size_cache)
    LOG.info(f"构建缓存完成, 用时 {time() - st:.2f} 秒")
    evaluated: List[Result] = []
    for workload, (indexes, counts) in zip(workloads, meta):
        conn = db_option.connections[workload.db]
        evaluated.append(_evaluate_workload(workload, indexes, counts, conn, real))

    return ModelResult(model_result.name, evaluated, model_result.id)


def evaluate_file(path: Path, args, db_option: DBOption, timestamp: str) -> Path:
    model_results = load_eval_results(path)
    if not model_results:
        raise ValueError("未找到模型评估结果")

    evaluated_results = []

    for mr in model_results:
        print(f"评估模型: {mr.name}")
        evaluated = evaluate_model_result(mr, db_option, args.workers, args.real)
        evaluated_results.append(evaluated)

    output_dir = Path(args.output_dir) if args.output_dir else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}.evaluated_{timestamp}{path.suffix}"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in evaluated_results], f, ensure_ascii=False, indent=4)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="使用真实数据库重新评估模型输出", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    DBOption.add_arg_group_to(parser)
    parser.add_argument("input_files", nargs="+", type=Path, help="eval.py 生成的结果 JSON 文件")
    parser.add_argument("--output-dir", type=Path, help="输出目录, 默认与输入文件相同")
    parser.add_argument("-w", "--workers", type=int, default=16, help="创建数据库缓存的进程数")
    parser.add_argument("--real", action="store_true", help="创建真实索引并执行查询")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    model_dbs = {db for db in args.db or [] if db}
    for path in args.input_files:
        for mr in load_eval_results(path):
            model_dbs.update({r.db for r in mr.results if r.db})
    if not model_dbs:
        raise ValueError("未提供数据库名称, 请使用 --db 或确保结果文件包含 db 字段")
    args.db = sorted(model_dbs)

    db_option = DBOption.from_args(args)
    log_to_file("eval_res", level="DEBUG" if args.debug else "INFO")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    outputs = []
    for path in args.input_files:
        print(f"处理文件: {path}")
        out_path = evaluate_file(path, args, db_option, timestamp)
        outputs.append(out_path)
        LOG.info(f"完成: {out_path}")

    LOG.info("全部完成:")
    for p in outputs:
        LOG.info(f"  {p}")


if __name__ == "__main__":
    main()

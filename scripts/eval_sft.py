import argparse
import json
import multiprocessing as mp
from dataclasses import asdict
from functools import partial
from pathlib import Path

from index_advisor import compute_reward
from index_advisor.db import (
    CostCache,
    DBOption,
    SizeCache,
    create_hypo_index_and_get_cost,
    create_index_and_get_cost,
    get_hypo_index_size,
    resolve_size_limits,
)
from index_advisor.eval import IndexRes, ModelResult, Result
from index_advisor.ia_logging import log_to_file, logger
from index_advisor.workload import Workload, load_workloads


logger = logger.getChild("eval_sft")


def eval_sft_workload(
    workload: Workload,
    db_option: DBOption,
    real: bool,
    cost_cache: CostCache,
    count_limit: int | None,
    size_limits: dict[str, float | None],
    size_cache: SizeCache,
) -> Result:
    """
    评估单个 workload 的索引推荐结果

    Args:
        workload: 工作负载对象，包含 queries 和 labels (推荐索引)
        db_option: 数据库连接选项
        real: 是否创建真实索引进行评估
        cost_cache: 成本缓存对象
        count_limit: 单个 workload 可评估的索引最大数量，None 表示不限制
        size_limit: 单个 workload 可评估的索引总大小上限 (Bytes)，None 表示不限制
        size_cache: 索引大小缓存对象

    Returns:
        评估结果 Result 对象
    """
    # 获取推荐的索引列表
    indexes = list(workload.labels) if workload.labels else []
    logger.debug(f"评估索引: {indexes}")

    # 使用数据库连接
    conn = db_option.connections[workload.db]

    original_count = len(indexes)
    limited_indexes = indexes

    indexes = limited_indexes
    if len(indexes) != original_count:
        logger.debug(f"根据限制截断索引: {original_count} -> {len(indexes)}")

    # 计算无索引的基础成本
    workload.basic_cost_explain = create_hypo_index_and_get_cost(conn, [], workload.queries, cost_cache)
    logger.debug(f"🏷️ 无索引 cost: {workload.basic_cost_explain}")

    limit_for_db = size_limits.get(workload.db)
    total_size = 0.0
    prev_cost = workload.basic_cost_explain
    selected = []
    index_res = []
    for idx in indexes[:count_limit]:
        size = get_hypo_index_size(conn.cursor(), workload.db, idx, size_cache)
        if limit_for_db is not None and total_size + size > limit_for_db:
            break
        total_size += size
        selected.append(idx)
        cur_cost = create_hypo_index_and_get_cost(conn, selected, workload.queries, cost_cache)
        profit = prev_cost - cur_cost
        prev_cost = cur_cost
        index_res.append(IndexRes(str(idx), 0, size=size, total_size=total_size, profit=profit))

    # 使用假设索引计算 cost
    actual_cost_hypo = create_hypo_index_and_get_cost(conn, indexes, workload.queries, cost_cache)
    logger.debug(f"🏷️ 假设索引 cost: {actual_cost_hypo}")

    # 使用真实索引计算 cost
    actual_cost_real = None
    if real:
        actual_cost_real = create_index_and_get_cost(conn, indexes, workload.queries, True, cost_cache)
        logger.debug(f"🏷️ 真实索引 cost: {actual_cost_real}")
        workload.basic_cost_real = create_index_and_get_cost(conn, [], workload.queries, True, cost_cache)
        logger.debug(f"🏷️ 无索引真实 cost: {workload.basic_cost_real}")

    # 计算 reward
    reward = compute_reward(
        workload.basic_cost_explain or workload.basic_cost,
        actual_cost_real or actual_cost_hypo,
        total=1,
        invalid=0,
        exist=0,
        duplicate=0,
    )
    if total_size > 0:
        reward /= total_size / (1024 * 1024 * 1024)
    else:
        reward = 0.0
    logger.debug(f"🎯 计算 reward: {reward}")

    return Result(
        workload_id=workload.id,
        db=workload.db,
        queries=[q.sql for q in workload.queries],
        sqls_hash=workload.sqls_hash,
        indexes=index_res,
        basic_cost_explain=workload.basic_cost_explain,
        basic_cost_real=workload.basic_cost_real,
        actual_cost_hypo=actual_cost_hypo,
        actual_cost_real=actual_cost_real,
        reward=reward,
        prompt="",
        response="",
        invalid=0,
        exist=0,
        duplicate=0,
        useless=0,
    )


def evaluate_workloads(workloads: list[Workload], args, size_limit_map: dict[str, float | None]) -> list[Result]:
    # 初始化数据库连接
    db_option = DBOption.from_args(args)
    cost_cache = CostCache()
    size_cache = SizeCache()
    results = []
    for i, workload in enumerate(workloads):
        logger.info(f"评估工作负载 {i + 1}/{len(workloads)}: ID={workload.id}")
        try:
            result = eval_sft_workload(
                workload,
                db_option,
                args.real,
                cost_cache,
                args.count_limit,
                size_limit_map,
                size_cache,
            )
            results.append(result)
            logger.info(f"完成评估，reward={result.reward:.4f}")
        except Exception as e:
            logger.error(f"评估工作负载 {workload.id} 时出错: {e}", exc_info=True)
            continue

    logger.info(f"评估完成，共处理 {len(results)} 个工作负载")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="评估 SFT 数据集中的索引推荐结果",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 添加数据库参数组
    DBOption.add_arg_group_to(parser)

    parser.add_argument("input_file", help="带 labels 的 workload JSON 文件路径")
    parser.add_argument("output_file", help="评估结果输出路径")
    parser.add_argument("--worker", help="并行工作进程数", type=int, default=32)
    parser.add_argument("--desc", help="评估任务的描述", type=str, default="评估SFT标签")
    parser.add_argument("--size-limit", "-s", help="单个 workload 推荐索引总大小上限 (MB)", type=float, default=None)
    parser.add_argument("--size-factor", help="按数据库体积计算索引大小上限", type=float, default=None)
    parser.add_argument("--count-limit", "-n", help="单个 workload 推荐索引数量上限", type=int, default=None)
    parser.add_argument("-r", "--real", help="创建真实索引并实际执行查询", action="store_true")
    parser.add_argument("--debug", help="调试模式", action="store_true")

    args = parser.parse_args()

    if args.count_limit is not None and args.count_limit <= 0:
        parser.error("--count-limit 必须大于 0")
    if args.size_limit is not None and args.size_limit <= 0:
        parser.error("--size-limit 必须大于 0")
    if args.size_factor is not None and args.size_factor <= 0:
        parser.error("--size-factor 必须大于 0")
    if args.size_limit is not None and args.size_factor is not None:
        parser.error("不能同时指定 --size-limit 与 --size-factor")

    # 设置日志级别
    log_to_file("eval_sft", "INFO" if not args.debug else "DEBUG")

    # 加载工作负载数据
    logger.info(f"从 {args.input_file} 加载工作负载数据...")
    workloads = load_workloads(args.input_file)
    logger.info(f"加载了 {len(workloads)} 个工作负载")

    if not workloads:
        logger.error("没有加载到有效的工作负载数据")
        return

    # 如果只有一个工作进程，直接顺序处理
    db_option = DBOption.from_args(args)
    size_limit_map = resolve_size_limits(db_option, args.size_limit, args.size_factor)
    # 释放连接, 后续每个 worker 会自行建立
    seen_conn: set[int] = set()
    for conn in db_option.connections.values():
        cid = id(conn)
        if cid in seen_conn:
            continue
        conn.close()
        seen_conn.add(cid)

    if args.worker == 1:
        results = evaluate_workloads(workloads, args, size_limit_map)
    else:
        # 将 workloads 划分成多个批次
        chunk_size = max(1, len(workloads) // args.worker)
        workload_chunks = [workloads[i : i + chunk_size] for i in range(0, len(workloads), chunk_size)]

        logger.info(f"将 {len(workloads)} 个工作负载划分为 {len(workload_chunks)} 个批次进行并行处理")

        with mp.get_context("spawn").Pool(processes=args.worker) as pool:
            fn = partial(evaluate_workloads, args=args, size_limit_map=size_limit_map)
            chunk_results = pool.map(fn, workload_chunks)

        # 展平结果列表
        results = [res for sublist in chunk_results for res in sublist]
        results.sort(key=lambda r: r.workload_id)

    # 计算总体统计
    if results:
        avg_reward = sum(r.reward for r in results) / len(results)
        avg_improvement_hypo = sum(r.cost_improvement_hypo for r in results) / len(results)
        logger.info(f"平均 reward: {avg_reward:.4f}")
        logger.info(f"平均假设索引改进: {avg_improvement_hypo:.4f} ({avg_improvement_hypo * 100:.2f}%)")

        if args.real and any(r.cost_improvement_real is not None for r in results):
            real_improvements = [r.cost_improvement_real for r in results if r.cost_improvement_real is not None]
            avg_improvement_real = sum(real_improvements) / len(real_improvements)
            logger.info(f"平均真实索引改进: {avg_improvement_real:.4f} ({avg_improvement_real * 100:.2f}%)")

    # 创建模型结果对象
    model_result = ModelResult(name="sft_labels", results=results)

    # 保存结果
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([asdict(model_result)], f, ensure_ascii=False, indent=4)

    logger.info(f"评估结果已保存到 {output_file}")


if __name__ == "__main__":
    main()

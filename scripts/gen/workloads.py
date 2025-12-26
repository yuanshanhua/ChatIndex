import os
from pathlib import Path

from index_advisor.dataset import load_queries_from_file
from index_advisor.db import DBOption
from index_advisor.ia_logging import log_to_file, logger
from index_advisor.workload import generate_workloads, save_workloads


logger = logger.getChild("gen")


def run():
    import argparse

    parser = argparse.ArgumentParser(
        description="从包含查询的 SQL 生成 workload 文件. 注意: 每次只能处理对同一数据库的查询集, 此数据库由 --db 参数指定.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    DBOption.add_arg_group_to(parser)
    parser.add_argument("input_file", help="输入 SQL 文件路径")
    parser.add_argument("output_dir", help="输出目录")
    parser.add_argument("name", help="数据集名称")
    parser.add_argument("--desc", help="任务描述", type=str, default="生成数据集")
    parser.add_argument("--debug", help="调试模式", action="store_true")
    parser.add_argument("-c", "--cost", help="解释查询获取基准 cost", action="store_true")
    parser.add_argument("-e", "--execute", help="执行查询获取基准 cost", action="store_true")
    parser.add_argument("--min-cost", "--mc", help="最小 cost. 低于此的查询将被过滤", type=float, default=-1)
    parser.add_argument("-n", "--num-workloads", help="生成的 prompt (工作负载) 数量", type=int, default=100)
    parser.add_argument("--min-size", "--ms", help="workload 中最小查询数量", type=int, default=5)
    parser.add_argument("--max-size", "--mss", help="workload 中最大查询数量", type=int, default=20)
    parser.add_argument("--max-frequency", help="workload 中查询的最大频率", type=int, default=1)

    args = parser.parse_args()

    if not args.cost and not args.execute and args.min_cost >= 0:
        raise ValueError("指定 --min-cost 时, 必须启用 --cost 或 --execute 之一")

    input_file = Path(args.input_file)
    workload_file = Path(args.output_dir) / f"{args.name}.workload.json"
    opt = DBOption.from_args(args)

    log_to_file("gen", "DEBUG" if args.debug else "INFO", args.desc)

    queries = load_queries_from_file(input_file, opt, args.cost, args.execute)
    if args.min_cost > 0:
        queries = {q for q in queries if q.basic_cost > args.min_cost}
    logger.info(f"筛选出 {len(queries)} 条 basic cost > {args.min_cost} 的 SQL")
    workloads = generate_workloads(
        opt.db, queries, args.num_workloads, args.min_size, args.max_size, args.max_frequency
    )

    os.makedirs(workload_file.parent, exist_ok=True)
    save_workloads(workload_file, workloads)


if __name__ == "__main__":
    run()

from pathlib import Path

from index_advisor.db import DBOption
from index_advisor.ia_logging import log_to_file, logger
from index_advisor.workload import generate_workloads, load_workloads, save_workloads


logger = logger.getChild("merge")


def run():
    import argparse

    parser = argparse.ArgumentParser(
        description="使用指定 workload 文件中出现的查询进行重组, 生成新的 workload 文件. 注意: 输入文件中所有 workload 必须来自同一数据库, 且与 --db 参数一致.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_file", help="输入 workload 文件路径", type=str)
    parser.add_argument("output", help="输出文件路径")
    parser.add_argument("--desc", help="任务描述", type=str, default="生成数据集")
    parser.add_argument("--debug", help="调试模式", action="store_true")
    parser.add_argument("-n", "--num-workloads", help="生成的工作负载数量", type=int, default=100)
    parser.add_argument("--min-size", "--ms", help="workload 中最小查询数量", type=int, default=5)
    parser.add_argument("--max-size", "--mss", help="workload 中最大查询数量", type=int, default=20)
    parser.add_argument("--max-frequency", help="workload 中查询的最大频率", type=int, default=1)
    DBOption.add_arg_group_to(parser)

    args = parser.parse_args()
    opt = DBOption.from_args(args)

    log_to_file("merge", "DEBUG" if args.debug else "INFO", args.desc)

    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"输入文件 {input_file} 不存在")
        exit(1)

    logger.info(f"加载工作负载文件 {input_file}")
    workloads = load_workloads(input_file)
    assert all(w.db == opt.db for w in workloads), "所有输入文件中的 workload 必须来自同一数据库, 且与 --db 参数一致"
    queries = {q for w in workloads for q in w.queries}
    logger.info(f"共加载 {len(workloads)} 个工作负载, 包含 {len(queries)} 个不同的查询")

    # 生成新的工作负载
    workloads = generate_workloads(
        opt.db, queries, args.num_workloads, args.min_size, args.max_size, args.max_frequency
    )

    # 保存合并后的工作负载
    if not args.output.endswith(".json"):
        args.output += ".json"

    workload_file = Path(args.output)
    save_workloads(workload_file, workloads)


if __name__ == "__main__":
    run()

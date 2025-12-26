from pathlib import Path

from index_advisor.ia_logging import log_to_file, logger
from index_advisor.workload import load_workloads, save_workloads


logger = logger.getChild("gen")


def run():
    import argparse

    parser = argparse.ArgumentParser(
        description="加载 workload 文件, 重新生成计算字段并保存.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_files", help="输入 workload 文件路径, 使用空格或 glob 模式指定多个文件", nargs="+")
    parser.add_argument("-s", "--suffix", help="输出后缀. 不指定则覆盖源文件", default=None)
    parser.add_argument("--explain", help="刷新 explain cost", action="store_true", default=False)
    parser.add_argument("--execute", help="刷新实际执行 cost", action="store_true", default=False)
    parser.add_argument("--debug", help="调试模式", action="store_true")
    # DBOption.add_arg_group_to(parser)
    args = parser.parse_args()
    # db_option = DBOption.from_args(args)

    log_to_file("refresh", "DEBUG" if args.debug else "INFO")

    for input_file in args.input_files:
        input_file = Path(input_file)
        if not input_file.exists():
            logger.error(f"输入文件 {input_file} 不存在")
            continue
        if args.suffix:
            output_file = input_file.with_stem(input_file.stem + args.suffix)
        else:
            output_file = input_file
        output_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"加载工作负载文件 {input_file}")
        w = load_workloads(input_file)
        # w = [rebuild(workload, db_option, args.explain, args.execute) for workload in w]
        save_workloads(output_file, w)


if __name__ == "__main__":
    run()

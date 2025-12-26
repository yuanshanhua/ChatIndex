import random
from pathlib import Path

from index_advisor.ia_logging import log_to_file, logger
from index_advisor.workload import Workload, load_workloads, save_workloads


logger = logger.getChild("merge")


def run():
    import argparse

    parser = argparse.ArgumentParser(
        description="合并多个 workload 文件, 随机排序并重新编码 ID.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_files", help="输入 workload 文件路径, 使用空格或 glob 模式指定多个文件", nargs="+")
    parser.add_argument("output", help="输出文件路径")
    parser.add_argument("--desc", help="任务描述", type=str, default="生成数据集")
    parser.add_argument("--debug", help="调试模式", action="store_true")
    args = parser.parse_args()

    if not args.input_files:
        logger.error("未指定输入文件，请使用 -i 或 --input-files 参数指定")
        return

    log_to_file("merge", "DEBUG" if args.debug else "INFO", args.desc)

    workloads: list[Workload] = []
    for input_file in args.input_files:
        input_file = Path(input_file)
        if not input_file.exists():
            logger.error(f"输入文件 {input_file} 不存在")
            continue

        logger.info(f"加载工作负载文件 {input_file}")
        workloads.extend(load_workloads(input_file))
    logger.info(f"共加载 {len(workloads)} 个工作负载")

    # 随机打乱工作负载顺序
    random.seed(0)
    random.shuffle(workloads)
    # 重新编码 ID
    for i, workload in enumerate(workloads):
        workload.id = i

    # 保存合并后的工作负载
    if not args.output.endswith(".json"):
        args.output += ".json"

    workload_file = Path(args.output)
    save_workloads(workload_file, workloads)


if __name__ == "__main__":
    run()

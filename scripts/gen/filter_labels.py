from pathlib import Path

from index_advisor.ia_logging import log_to_file, logger
from index_advisor.workload import Workload, load_workloads, save_workloads


logger = logger.getChild("filter_labels")


def run():
    import argparse

    parser = argparse.ArgumentParser(
        description="剔除 labels 数量少于指定阈值的 workload.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("workload_file", help="输入 workload 文件路径")
    parser.add_argument("min_labels", type=int, help="保留 labels 数量不少于该值的 workload")
    parser.add_argument(
        "-o",
        "--output",
        help="输出 workload 文件路径. 默认在原文件名后附加 labels-ge{min_labels} 后缀",
    )
    parser.add_argument("--desc", help="任务描述", type=str, default="过滤 workload")
    parser.add_argument("--debug", help="调试模式", action="store_true")
    args = parser.parse_args()

    if args.min_labels < 0:
        parser.error("min_labels 不能小于 0")

    log_to_file("filter_labels", "DEBUG" if args.debug else "INFO", args.desc)

    workload_path = Path(args.workload_file)
    if not workload_path.exists():
        logger.error("输入文件不存在: %s", workload_path)
        return

    workloads = load_workloads(workload_path)
    total = len(workloads)

    min_labels = args.min_labels
    kept: list[Workload] = []
    removed = 0
    for workload in workloads:
        label_count = len(workload.labels) if workload.labels else 0
        if label_count >= min_labels:
            kept.append(workload)
            continue
        removed += 1
        logger.info("移除 workload %s (hash: %s), labels 数量为 %s", workload.id, workload.sqls_hash, label_count)

    if args.output:
        output_path = Path(args.output)
    else:
        suffix = workload_path.suffix or ".json"
        output_name = f"{workload_path.stem}.labels-ge{min_labels}{suffix}"
        output_path = workload_path.with_name(output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("原始 workload 数量: %s", total)
    logger.info("保留 workload 数量: %s", len(kept))
    logger.info("被剔除 workload 数量: %s", removed)

    save_workloads(output_path, kept)
    logger.info("输出文件: %s", output_path)


if __name__ == "__main__":
    run()

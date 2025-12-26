import json
from pathlib import Path

from index_advisor.ia_logging import log_to_file, logger
from index_advisor.msg import to_data_sample
from index_advisor.workload import Workload, load_workloads


logger = logger.getChild("gen")


def run():
    import argparse

    parser = argparse.ArgumentParser(
        description="加载 workload 文件并生成 prompt. 此脚本已无用, 现在训练不需要预生成 prompt 文件, 仅供检查 prompt 格式使用.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("workload", help="workload 文件路径")
    parser.add_argument("--name", help="输出文件名. 全名为 {name}.prompt.json", default="")
    parser.add_argument("--desc", help="任务描述", type=str, default="生成 prompt")
    parser.add_argument("--max-len", help="限制单个 workload 的最大 SQL 长度", type=int, default=-1)
    parser.add_argument("--start", help="workload 起始索引", type=int, default=0)
    parser.add_argument("--end", help="workload 结束索引", type=int, default=-1)
    parser.add_argument("--mm", help="启用 column 多模态", action="store_true")
    parser.add_argument("--mm2", help="启用 sql 多模态", action="store_true")
    parser.add_argument("--debug", help="调试模式", action="store_true")
    args = parser.parse_args()

    workload_file = Path(args.workload)
    name = workload_file.stem.replace(".workload", "") if not args.name else args.name
    output_file = workload_file.parent / f"{name}.prompt.json"

    log_to_file("gen_prompts", "DEBUG" if args.debug else "INFO", args.desc)

    workloads = load_workloads(workload_file)
    logger.info(f"共加载 {len(workloads)} 个 workload")
    if args.max_len:
        max_len = int(args.max_len)
        if max_len > 0:
            workloads = [w for w in workloads if w.total_sql_length <= max_len]
            logger.info(f"限制最大 SQL 长度 {max_len}, 过滤后剩余 {len(workloads)} 个 workload")

    # show workload statistics
    if args.end == -1:
        args.end = len(workloads)
    workloads = workloads[args.start : args.end]
    save_prompts(output_file, workloads, args.max_len, args.mm, args.mm2)


def save_prompts(file: str | Path, workloads: list[Workload], max_len: int, column_mm: bool, sql_mm: bool):
    dataset = [to_data_sample(w, column_mm, sql_mm) for w in workloads]
    if max_len > 0:
        dataset = [d for d in dataset if len(d["instruction"]) <= max_len]
        logger.info(f"限制最大 prompt 长度 {max_len}")
    # dataset statistics
    logger.info(f"共生成 {len(dataset)} 条 prompt")
    max_length = max(len(d["instruction"]) for d in dataset)
    logger.info(f"最大 prompt 长度: {max_length}")
    min_length = min(len(d["instruction"]) for d in dataset)
    logger.info(f"最小 prompt 长度: {min_length}")
    avg_length = sum(len(d["instruction"]) for d in dataset) / len(dataset)
    logger.info(f"平均 prompt 长度: {avg_length:.2f}")
    max_columns = max(len(w.column_ids) for w in workloads)
    logger.info(f"最大列数: {max_columns}")
    min_columns = min(len(w.column_ids) for w in workloads)
    logger.info(f"最小列数: {min_columns}")
    avg_columns = sum(len(w.column_ids) for w in workloads) / len(workloads)
    logger.info(f"平均列数: {avg_columns:.2f}")

    logger.info(f"写入 prompts 到 {str(file)}")
    with open(file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    run()

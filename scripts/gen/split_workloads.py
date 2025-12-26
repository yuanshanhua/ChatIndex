import argparse
from pathlib import Path

from index_advisor.ia_logging import logger
from index_advisor.workload import load_workloads, save_workloads


def main():
    parser = argparse.ArgumentParser(
        description="按数据库分割 workload 文件, 为每个数据库生成单独的 workload 文件.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_files", help="输入 workload 文件路径, 使用空格或 glob 模式指定多个文件", nargs="+")
    args = parser.parse_args()

    for input_file in args.input_files:
        input_file = Path(input_file)
        if not input_file.exists():
            logger.error(f"输入文件 {input_file} 不存在")
            continue
        logger.info(f"加载工作负载文件 {input_file}")
        w = load_workloads(input_file)

        db_workloads = {}
        for workload in w:
            db_workloads.setdefault(workload.db, []).append(workload)

        for db, workloads in db_workloads.items():
            output_file = input_file.parent / f"{db}.json"
            if output_file.exists():
                output_file = input_file.with_stem(f"{input_file.stem}_{db}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"保存数据库 {db} 的工作负载到文件 {output_file}")
            save_workloads(output_file, workloads)


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
from typing import List, Set

from index_advisor.ia_logging import log_to_file, logger


logger = logger.getChild("extract_workload_subset")


def normalize_sql_list(sqls: List[str]) -> Set[str]:
    """标准化SQL列表，去除空白字符并转为集合用于比较"""
    return {sql.strip() for sql in sqls if sql.strip()}


def load_target_workloads(target_file: str) -> List[Set[str]]:
    """
    从目标文件（dsb.7000q.1000w.1q.300c.workload.json）中加载所有workload的SQL集合

    返回：每个workload的SQL字符串集合列表
    """
    logger.info(f"从目标文件加载工作负载: {target_file}")

    with open(target_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_sql_sets = []
    for workload in data:
        # 从queries中提取SQL
        sqls = [query["sql"] for query in workload.get("queries", [])]
        sql_set = normalize_sql_list(sqls)
        if sql_set:  # 只添加非空的SQL集合
            target_sql_sets.append(sql_set)

    logger.info(f"已加载 {len(target_sql_sets)} 个目标工作负载")
    return target_sql_sets


def extract_matching_subset(source_file: str, target_sql_sets: List[Set[str]]) -> tuple[List[dict], int]:
    """
    从源文件（dsb_7000.extend.json）中提取匹配的workload子集

    返回：(匹配的workload列表, 源文件总数)
    """
    logger.info(f"从源文件加载工作负载: {source_file}")

    with open(source_file, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    matching_workloads = []

    for i, item in enumerate(source_data):
        if "extend" not in item:
            continue

        extend_data = item["extend"]
        if "workload" not in extend_data:
            continue

        # 提取并标准化SQL列表
        source_sqls = extend_data["workload"]
        source_sql_set = normalize_sql_list(source_sqls)

        # 检查是否与任何目标workload匹配
        for j, target_sql_set in enumerate(target_sql_sets):
            if source_sql_set == target_sql_set:
                logger.debug(f"找到匹配: source[{i}] 匹配 target[{j}] ({len(source_sql_set)} 个查询)")
                matching_workloads.append(item)
                break

    logger.info(f"找到 {len(matching_workloads)} 个匹配的工作负载")
    return matching_workloads, len(source_data)


def main():
    parser = argparse.ArgumentParser(
        description="从 index eab 的结果中提取仅包含指定 Workload 文件的子集. 当 index eab 可以直接处理 Workload 文件后可废弃"
    )
    parser.add_argument("--source", default="workloads/sft/dsb_7000.extend.json", help="index eab 结果文件")
    parser.add_argument(
        "--target", default="workloads/index_eab/1q/dsb.7000q.1000w.1q.300c.workload.json", help="Workload 文件"
    )
    parser.add_argument("--output", default="workloads/sft/dsb_7000.extend.subset.json", help="输出文件")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", help="调试模式", action="store_true")

    args = parser.parse_args()

    # 设置日志级别
    log_to_file("extract_workload_subset", "INFO" if not args.debug else "DEBUG")

    # 检查输入文件是否存在
    source_path = Path(args.source)
    target_path = Path(args.target)

    if not source_path.exists():
        logger.error(f"源文件未找到: {source_path}")
        return 1

    if not target_path.exists():
        logger.error(f"目标文件未找到: {target_path}")
        return 1

    try:
        # 加载目标workloads
        target_sql_sets = load_target_workloads(str(target_path))

        if not target_sql_sets:
            logger.warning("未找到有效的目标工作负载")
            return 1

        # 提取匹配的子集
        matching_workloads, source_count = extract_matching_subset(str(source_path), target_sql_sets)

        if not matching_workloads:
            logger.warning("未找到匹配的工作负载")
            return 1

        # 保存结果
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(matching_workloads, f, indent=2, ensure_ascii=False)

        logger.info(f"成功提取 {len(matching_workloads)} 个匹配的工作负载")
        logger.info(f"输出已保存到: {output_path}")

        # 统计信息
        if args.verbose:
            logger.info("统计信息:")
            logger.info(f"  源工作负载数: {source_count}")
            logger.info(f"  目标工作负载数: {len(target_sql_sets)}")
            logger.info(f"  匹配工作负载数: {len(matching_workloads)}")

            if matching_workloads:
                # 显示一些匹配的workload信息
                logger.info("匹配工作负载示例:")
                for i, workload in enumerate(matching_workloads[:3]):
                    extend_data = workload.get("extend", {})
                    sql_count = len(extend_data.get("workload", []))
                    index_count = len(extend_data.get("indexes", []))
                    logger.info(f"  [{i + 1}] {sql_count} 个查询, {index_count} 个索引")

        return 0

    except Exception as e:
        logger.error(f"执行出错: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

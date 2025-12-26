import argparse
import json
import sys
from dataclasses import dataclass
from typing import List, Optional

from index_advisor.eval import ModelResult, Result, load_eval_results
from index_advisor.schemas import DatabaseSchema, Index, parse_indexes_str, schemas
from index_advisor.workload import Workload, load_workloads


@dataclass
class MatchResult:
    """单个结果的匹配信息"""

    workload_id: int
    sqls_hash: str
    has_labels: bool
    parsed_successfully: bool
    recommended_indexes: set[Index]
    label_indexes: set[Index]
    equal: bool
    matched_indexes: set[Index]
    unmatched_recommended: set[Index]
    unmatched_labels: set[Index]
    match_count: int
    total_recommended: int
    total_labels: int


@dataclass
class ModelMatchSummary:
    """模型匹配度汇总"""

    model_name: str
    checkpoint_id: int
    total_results: int
    results_with_labels: int
    results_without_labels: int
    parse_failures: int
    equal_count: int
    total_matches: int
    total_recommended: int
    total_labels: int
    match_results: List[MatchResult]


def find_workload_by_hash(workloads: List[Workload], sqls_hash: str) -> Optional[Workload]:
    """根据 sqls_hash 查找对应的 workload"""
    for workload in workloads:
        if workload.sqls_hash == sqls_hash:
            return workload
    return None


def compare_result_with_labels(result: Result, workload: Workload, db_schema: DatabaseSchema) -> MatchResult:
    """比较单个结果与对应工作负载的标签"""

    # 解析推荐的索引
    recommended_indexes = []
    parsed_successfully = True
    try:
        recommended_indexes, _ = parse_indexes_str(result.response, db_schema)
    except Exception:
        parsed_successfully = False

    # 获取标签索引
    label_indexes = set()
    has_labels = False
    if workload.labels:
        label_indexes = set(workload.labels)
        has_labels = True

    # 计算匹配情况
    recommended_indexes = set(recommended_indexes)
    matched_indexes = recommended_indexes & label_indexes
    unmatched_recommended = recommended_indexes - label_indexes
    unmatched_labels = label_indexes - recommended_indexes

    return MatchResult(
        workload_id=result.workload_id,
        sqls_hash=result.sqls_hash,
        has_labels=has_labels,
        parsed_successfully=parsed_successfully,
        recommended_indexes=recommended_indexes,
        label_indexes=label_indexes,
        equal=recommended_indexes == label_indexes,
        matched_indexes=matched_indexes,
        unmatched_recommended=unmatched_recommended,
        unmatched_labels=unmatched_labels,
        match_count=len(matched_indexes),
        total_recommended=len(recommended_indexes),
        total_labels=len(label_indexes),
    )


def analyze_model_result(model_result: ModelResult, workloads_by_hash: dict[str, Workload]) -> ModelMatchSummary:
    """分析单个模型结果的匹配度"""

    match_results = []
    equal_count = 0
    total_matches = 0
    total_recommended = 0
    total_labels = 0
    results_with_labels = 0
    results_without_labels = 0
    parse_failures = 0

    for result in model_result.results:
        # 查找对应的 workload
        w = workloads_by_hash.get(result.sqls_hash)
        if w is None:
            results_without_labels += 1
            continue

        # 比较结果
        match_result = compare_result_with_labels(result, w, schemas[w.db])
        match_results.append(match_result)

        # 统计信息
        if match_result.has_labels:
            results_with_labels += 1
        else:
            results_without_labels += 1

        if not match_result.parsed_successfully:
            parse_failures += 1

        if match_result.equal:
            equal_count += 1

        total_matches += match_result.match_count
        total_recommended += match_result.total_recommended
        total_labels += match_result.total_labels

    return ModelMatchSummary(
        model_name=model_result.name,
        checkpoint_id=model_result.id,
        total_results=len(model_result.results),
        results_with_labels=results_with_labels,
        results_without_labels=results_without_labels,
        parse_failures=parse_failures,
        equal_count=equal_count,
        total_matches=total_matches,
        total_recommended=total_recommended,
        total_labels=total_labels,
        match_results=match_results,
    )


def print_summary_report(summaries: List[ModelMatchSummary]):
    """打印汇总报告"""
    print("\n" + "=" * 80)
    print("索引推荐与标签匹配度评估报告")
    print("=" * 80)

    for summary in summaries:
        print(f"\nModel: {summary.model_name} (Checkpoint ID: {summary.checkpoint_id})")
        print("-" * 60)
        print(f"总结果数量:           {summary.total_results}")
        print(f"有标签的结果:         {summary.results_with_labels}")
        print(f"无标签的结果:         {summary.results_without_labels}")
        print(f"解析失败的结果:       {summary.parse_failures}")
        print(f"完全匹配的结果:       {summary.equal_count}")
        print(f"总推荐索引数:         {summary.total_recommended}")
        print(f"总标签索引数:         {summary.total_labels}")
        print(f"总匹配索引数:         {summary.total_matches}")

        # 计算百分比
        if summary.results_with_labels > 0:
            labels_coverage = summary.results_with_labels / summary.total_results * 100
            print(f"标签覆盖率:          {labels_coverage:.1f}%")

        if summary.equal_count > 0:
            recall = summary.equal_count / summary.results_with_labels * 100
            print(f"完全匹配占比:   {recall:.1f}%")


def export_detailed_results(summaries: List[ModelMatchSummary], output_file: str):
    """导出详细的匹配结果到JSON文件"""

    def index_to_dict(index: Index) -> dict:
        return {"table": index.table, "columns": index.columns}

    def match_result_to_dict(match_result: MatchResult) -> dict:
        return {
            "workload_id": match_result.workload_id,
            "sqls_hash": match_result.sqls_hash,
            "has_labels": match_result.has_labels,
            "parsed_successfully": match_result.parsed_successfully,
            "equal": match_result.equal,
            "recommended_indexes": [index_to_dict(idx) for idx in match_result.recommended_indexes],
            "label_indexes": [index_to_dict(idx) for idx in match_result.label_indexes],
            "matched_indexes": [index_to_dict(idx) for idx in match_result.matched_indexes],
            "unmatched_recommended": [index_to_dict(idx) for idx in match_result.unmatched_recommended],
            "unmatched_labels": [index_to_dict(idx) for idx in match_result.unmatched_labels],
            "match_count": match_result.match_count,
            "total_recommended": match_result.total_recommended,
            "total_labels": match_result.total_labels,
        }

    export_data = []
    for summary in summaries:
        model_data = {
            "model_name": summary.model_name,
            "checkpoint_id": summary.checkpoint_id,
            "summary": {
                "total_results": summary.total_results,
                "results_with_labels": summary.results_with_labels,
                "results_without_labels": summary.results_without_labels,
                "parse_failures": summary.parse_failures,
                "equal_count": summary.equal_count,
                "total_matches": summary.total_matches,
                "total_recommended": summary.total_recommended,
                "total_labels": summary.total_labels,
            },
            "detailed_results": [match_result_to_dict(mr) for mr in summary.match_results],
        }
        export_data.append(model_data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已导出到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="评估模型推荐索引与标签的匹配度")
    parser.add_argument("eval_file", help="评估结果JSON文件路径")
    parser.add_argument("labeled_workload", help="包含标签的工作负载文件路径")
    parser.add_argument("-o", "--output", help="详细结果输出JSON文件路径（可选）")

    args = parser.parse_args()

    try:
        # 加载评估结果
        model_results = load_eval_results(args.eval_file)
        if not model_results:
            print("错误: 没有找到评估结果")
            sys.exit(1)

        # 加载工作负载（带标签）
        workloads = load_workloads(args.labeled_workload)
        print(f"加载了 {len(workloads)} 个工作负载")
        workloads_by_hash = {w.sqls_hash: w for w in workloads}

        # 按checkpoint ID排序
        model_results.sort(key=lambda x: x.id)

        # 分析每个模型结果
        summaries = []
        for model_result in model_results:
            summary = analyze_model_result(model_result, workloads_by_hash)
            summaries.append(summary)

        # 打印汇总报告
        print_summary_report(summaries)

        # 导出详细结果（如果指定了输出文件）
        if args.output:
            export_detailed_results(summaries, args.output)

    except FileNotFoundError as e:
        print(f"错误: 找不到文件 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

import argparse
import statistics
import traceback
from bisect import bisect_right
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, cast

from index_advisor import compute_reward
from index_advisor.db import DBOption
from index_advisor.eval import ModelResult, Result, load_eval_results
from index_advisor.size_limit_utils import SizeScenario, build_size_scenarios
from index_advisor.utils import parse_ranges


def calculate_stats(values: List[float]) -> Dict[str, float]:
    """计算统计指标"""
    if not values:
        return {}

    return {
        "max": max(values),
        "min": min(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def calculate_checkpoint_stats(model_result: ModelResult) -> Dict[str, Dict[str, float]]:
    """计算单个checkpoint的统计信息，返回字段统计字典"""
    # 定义要统计的字段
    fields: list[tuple[str, Callable[[Result], float | int | None]]] = [
        ("Basic Cost (解释)", lambda r: r.basic_cost_explain),
        ("Basic Cost (执行)", lambda r: r.basic_cost_real),
        ("虚拟索引 Cost", lambda r: r.actual_cost_hypo),
        ("真实索引 Cost", lambda r: r.actual_cost_real),
        # ("SWIRL reward", lambda r: r.swirl_reward),
        # ("another reward", lambda r: r.another_reward),
        # ("reward", lambda r: r.reward),
        # ("真实 Cost 改进比例", lambda r: r.cost_improvement_real),
        # ("虚拟 Cost 改进比例", lambda r: r.cost_improvement_hypo),
        # ("无效", lambda r: r.invalid),
        # ("已存在", lambda r: r.exist),
        # ("重复", lambda r: r.duplicate),
        # ("无用", lambda r: r.useless),
    ]

    checkpoint_stats = {}
    for field, getter in fields:
        values = [getter(result) for result in model_result.results]
        # 过滤掉None值
        valid_values = [v for v in values if v is not None]

        if valid_values:
            stats = calculate_stats(cast(List[float], valid_values))
            if stats:
                checkpoint_stats[field] = stats

    return checkpoint_stats


def p(num: float) -> str:
    if num == 0:
        return "0"
    abs_num = abs(num)
    if abs_num < 1e-3:
        return f"{num:.3e}"
    elif abs_num >= 1000:
        return f"{num:,.2f}"
    else:
        return f"{num:.4g}"


def print_checkpoint_summary(model_result: ModelResult):
    """打印单个checkpoint的汇总信息"""
    print(f"\n{'=' * 60}")
    print(f"Checkpoint: {model_result.name} (ID: {model_result.id})")
    print(f"Total workloads: {len(model_result.results)}")
    print(f"{'=' * 60}")

    checkpoint_stats = calculate_checkpoint_stats(model_result)

    for field, stats in checkpoint_stats.items():
        print(f"\n{field}:")
        print(f"  最大值:    {p(stats['max'])}")
        print(f"  最小值:    {p(stats['min'])}")
        print(f"  平均值:    {p(stats['mean'])}")
        print(f"  中位数:    {p(stats['median'])}")
        print(f"  标准差:    {p(stats['std'])}")

    return checkpoint_stats


def print_checkpoint_rankings(model_results: List[ModelResult], title: str | None = None, top: int | None = None):
    """按各字段平均值排序打印checkpoint排名"""
    print(f"\n{'=' * 80}")
    print(title or "按各字段平均值排序的 Checkpoint 排名")
    print(f"{'=' * 80}")

    # 收集所有checkpoint的统计数据
    checkpoint_stats = {}
    for model_result in model_results:
        stats = calculate_checkpoint_stats(model_result)
        checkpoint_stats[model_result.name] = {"id": model_result.id, "stats": stats}

    # 获取所有可能的字段
    all_fields = set()
    for checkpoint_data in checkpoint_stats.values():
        all_fields.update(checkpoint_data["stats"].keys())

    # 为每个字段创建排名
    for field in sorted(all_fields):
        if "basic" in field.lower():
            # 基础数据只需打印一次, 取任意一个checkpoint的数据
            print(f"\n{field} = {p(next(iter(checkpoint_stats.values()))['stats'][field]['mean'])}")
            print("-" * 50)
            continue
        prefer_lower = "cost" in field.lower()  # 成本类字段数值越低越好
        order_label = "升序(更低更好)" if prefer_lower else "降序(更高更好)"
        print(f"\n{field.upper()} RANKING (按平均值{order_label}):")
        print("-" * 50)

        # 收集该字段的数据
        field_data = []
        for checkpoint_name, checkpoint_data in checkpoint_stats.items():
            if field in checkpoint_data["stats"]:
                mean_value = checkpoint_data["stats"][field]["mean"]
                field_data.append((checkpoint_name, checkpoint_data["id"], mean_value))

        # 按均值排序: 成本越低越好, 奖励越高越好
        field_data.sort(key=lambda x: x[2], reverse=not prefer_lower)

        # 仅展示前 top 名
        display_data = field_data if top is None or top <= 0 else field_data[:top]

        # 打印排名
        for rank, (name, checkpoint_id, mean_value) in enumerate(display_data, 1):
            print(f"  {rank:2d}. {name: >16} (ID: {checkpoint_id: >5}) - Mean: {p(mean_value)}")


class IndexTruncationPlanner:
    """Caches incremental truncation results for a workload result."""

    def __init__(self, result: Result):
        self.result = result
        self.indexes = list(result.indexes)
        self.cumulative_sizes: list[float] = []
        self.cumulative_profits: list[float] = []
        total_size = 0.0
        total_profit = 0.0
        for idx in self.indexes:
            total_size += idx.size or 0.0
            self.cumulative_sizes.append(total_size)
            total_profit += idx.profit or 0.0
            self.cumulative_profits.append(total_profit)
        self.cache: dict[float, tuple[Result, bool]] = {}

    def for_limit(self, limit_bytes: float | None) -> tuple[Result, bool]:
        if limit_bytes is None or not self.indexes:
            return self.result, False
        if limit_bytes in self.cache:
            return self.cache[limit_bytes]
        idx_count = bisect_right(self.cumulative_sizes, limit_bytes)
        if idx_count >= len(self.indexes):
            res = (self.result, False)
            self.cache[limit_bytes] = res
            return res
        truncated = self._build_truncated(idx_count)
        res = (truncated, True)
        self.cache[limit_bytes] = res
        return res

    def _build_truncated(self, count: int) -> Result:
        total_size = self.cumulative_sizes[count - 1] if count else 0.0
        total_profit = self.cumulative_profits[count - 1] if count else 0.0
        truncated_indexes = list(self.indexes[:count])
        response = "\n".join(idx.index for idx in truncated_indexes)
        actual_cost_hypo = max(self.result.basic_cost_explain - total_profit, 0.0)
        basic_cost = self.result.basic_cost_real or self.result.basic_cost_explain
        reward_raw = compute_reward(
            basic_cost,
            actual_cost_hypo,
            total=1,
            invalid=self.result.invalid,
            exist=self.result.exist,
            duplicate=self.result.duplicate,
        )
        total_size_mb = total_size / (1024 * 1024) if total_size else 0.0
        normalized_reward = reward_raw / total_size_mb if total_size_mb > 0 else 0.0
        return replace(
            self.result,
            indexes=truncated_indexes,
            response=response,
            actual_cost_real=None,
            actual_cost_hypo=actual_cost_hypo,
            reward=normalized_reward,
        )


def apply_size_limits_to_model_results(
    model_results: List[ModelResult],
    scenario: SizeScenario,
    planner_cache: Dict[int, IndexTruncationPlanner],
) -> tuple[List[ModelResult], int]:
    """Return cloned model results adjusted for the specified scenario."""

    truncated_workloads = 0
    adjusted_results: List[ModelResult] = []
    for model_result in model_results:
        new_results: List[Result] = []
        for result in model_result.results:
            key = id(result)
            planner = planner_cache.setdefault(key, IndexTruncationPlanner(result))
            limit = scenario.limits.get(result.db or "")
            truncated_result, truncated = planner.for_limit(limit)
            if truncated:
                truncated_workloads += 1
            new_results.append(truncated_result)
        adjusted_results.append(ModelResult(model_result.name, new_results, model_result.id))
    return adjusted_results, truncated_workloads


def filter_model_results_by_db(model_results: List[ModelResult], databases: List[str]) -> int:
    """按数据库名称过滤 workloads, 返回被过滤掉的数量."""
    if not databases:
        return 0

    targets = {db.lower() for db in databases}
    removed = 0
    for model_result in model_results:
        original_count = len(model_result.results)
        model_result.results = [r for r in model_result.results if (r.db or "").lower() in targets]
        removed += original_count - len(model_result.results)
    return removed


def main():
    parser = argparse.ArgumentParser(description="生成评估结果汇总报告")
    DBOption.add_arg_group_to(parser)
    parser.add_argument("input_files", nargs="+", type=Path, help="评估结果 JSON 文件路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="启用详细输出")
    parser.add_argument(
        "--size-limit",
        "-s",
        help="单个 workload 推荐索引大小上限 (MB), 可指定多个值以空格分隔",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--size-factor",
        "-f",
        help="按数据库体积因子设定索引大小上限, 可指定多个值",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.5, 1.0],
    )
    parser.add_argument(
        "--filter-db",
        "-d",
        nargs="+",
        help="仅统计来自指定数据库的 workload 的评估结果",
        default=["tpch", "tpcds", "dsb", "imdb", "gifshow"],
    )
    parser.add_argument("--top", "-t", type=int, help="打印前 N 名", default=10)
    parser.add_argument("--ckpt", help="指定要读取的 checkpoint 序号, 逗号分隔或 start:end:step", default=None)
    args = parser.parse_args()
    if args.size_limit and args.size_factor:
        args.size_factor = None
        print("注意: 同时指定了 --size-limit 和 --size-factor, 将忽略 --size-factor 参数")

    for path, db in product(args.input_files, args.filter_db):
        print(f"\n{'#' * 80}")
        print("#" * 80)
        print(f"处理文件: {path}, 数据库过滤: {db or '无'}")
        print("#" * 80)
        print("#" * 80)
        try:
            handle_file(path, db, args)
        except FileNotFoundError:
            print(f"错误: 找不到文件 {path}")
            continue
        except Exception as e:
            print(f"错误: {e}")
            traceback.print_exc()
            continue


def handle_file(path: Path, db: str, args):
    # 加载评估结果
    model_results = load_eval_results(path)

    if not model_results:
        print("没有找到评估结果")
        return

    if db:
        removed = filter_model_results_by_db(model_results, [db])
        if removed:
            print(f"根据数据库筛选移除 {removed} 个 workload, 保留数据库: {db}")
        else:
            print(f"数据库筛选条件未匹配任何 workload: {', '.join(args.filter_db)}")

    if args.ckpt:
        ckpt_ids = parse_ranges(args.ckpt)
        original_count = len(model_results)
        model_results = [mr for mr in model_results if mr.id in ckpt_ids]
        print(f"根据 checkpoint 序号筛选: 原有 {original_count} 个, 保留 {len(model_results)} 个")

    all_db = {r.db for mr in model_results for r in mr.results if r.db}
    if args.db:
        all_db.update(args.db)
    args.db = list(all_db)
    print(f"自动添加数据库: {', '.join(args.db)}")

    db_option = DBOption.from_args(args)
    scenarios = build_size_scenarios(db_option, args.size_limit, args.size_factor)
    planner_cache: Dict[int, IndexTruncationPlanner] = {}

    print("索引推荐结果评估报告")
    print(f"checkpoint 数量: {len(model_results)}")
    print(f"读取评估结果: {path}")

    for scenario in scenarios:
        print(f"\n{'#' * 80}")
        print(f"场景: {scenario.label}")
        print(f"{'#' * 80}")

        scenario_results, truncated = apply_size_limits_to_model_results(model_results, scenario, planner_cache)
        if truncated:
            print(f"按照大小限制截断 {truncated} 个 workload 的索引")
        else:
            print("大小限制未触发截断")

        ordered_results = sorted(scenario_results, key=lambda x: x.id)
        if args.verbose:
            for model_result in ordered_results:
                print_checkpoint_summary(model_result)

        print_checkpoint_rankings(
            ordered_results,
            title=f"{scenario.label} - 按各字段平均值排序的 Checkpoint 排名",
            top=args.top,
        )


if __name__ == "__main__":
    main()

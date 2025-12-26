"""Analyze workload or prompt JSON files and optionally inspect token stats."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from index_advisor.mm.model import MMConfig
from index_advisor.workload import load_workloads
from lmf_hooks.model import hook_tokenizer, mm_init
from scripts.sft_projector import CustomDataset


TokenStatsConfig = dict[str, float]


_TOKENIZER: Any | None = None
_TOKENIZER_PATH: Optional[str] = None
_ACTIVE_MM_CONFIG: Optional[MMConfig] = None


def _passthrough_collate(batch: List[dict[str, Any]]) -> List[dict[str, Any]]:
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分析 workload 或 prompt JSON 文件并可选统计 token 分布",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("files", nargs="+", help="待分析的 JSON 文件路径")
    parser.add_argument("-t", "--tokenizer", action="store_true", help="启用 token 统计功能，需提供模型名称或路径")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="models/qwen2.5-7b-instruct",
        help="当提供模型名称或路径时, 使用 CustomDataset 统计 token 分布",
    )
    parser.add_argument("--group-size", "--gs", type=int, default=1, help="构造多轮样本时的 group_size, 默认为 1")
    parser.add_argument(
        "--token-sample-limit", type=int, default=None, help="限制用于 token 统计的样本数量, 默认使用全部"
    )
    parser.add_argument("--token-batch-size", type=int, default=8, help="token 统计时 DataLoader 的 batch_size")
    parser.add_argument("--token-num-workers", type=int, default=16, help="token 统计 DataLoader 使用的工作进程数")
    MMConfig.add_arg_group_to(parser)
    args = parser.parse_args()
    if args.group_size <= 0:
        parser.error("--group-size 必须为正整数")
    if args.token_sample_limit is not None and args.token_sample_limit < 0:
        parser.error("--token-sample-limit 不能为负数")
    if args.token_batch_size <= 0:
        parser.error("--token-batch-size 必须为正整数")
    if args.token_num_workers is not None and args.token_num_workers < 0:
        parser.error("--token-num-workers 不能为负数")
    return args


def detect_file_type(data: Any) -> str:
    """
    检测 JSON 文件类型

    Returns:
        'workload': workload.json 文件
        'prompt': prompt.json 文件
        'unknown': 未知类型
    """
    if isinstance(data, dict):
        return "workload"

    if not data:
        return "unknown"

    first_item = data[0]
    if not isinstance(first_item, dict):
        return "unknown"

    # 检查是否为 prompt 文件 (包含 input, output, instruction)
    prompt_keys = {"input", "output", "instruction"}
    if prompt_keys.issubset(first_item.keys()):
        return "prompt"

    return "workload"


def get_tokenizer(model_path: str, mm_config: MMConfig) -> tuple[Any, MMConfig]:
    global _TOKENIZER, _TOKENIZER_PATH, _ACTIVE_MM_CONFIG
    if _TOKENIZER is None or _TOKENIZER_PATH != model_path or _ACTIVE_MM_CONFIG is not mm_config:
        mm_init(mm_config)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer = hook_tokenizer(tokenizer)
        _TOKENIZER = tokenizer
        _TOKENIZER_PATH = model_path
        _ACTIVE_MM_CONFIG = mm_config
    else:
        mm_init(mm_config)
    assert _TOKENIZER is not None and _ACTIVE_MM_CONFIG is not None
    return _TOKENIZER, _ACTIVE_MM_CONFIG


def summarize_distribution(values: List[int]) -> Optional[TokenStatsConfig]:
    if not values:
        return None
    sorted_vals = sorted(values)
    count = len(sorted_vals)

    def percentile(p: float) -> float:
        if count == 1:
            return float(sorted_vals[0])
        k = (count - 1) * p / 100.0
        lower = int(k)
        upper = min(lower + 1, count - 1)
        weight = k - lower
        return sorted_vals[lower] + (sorted_vals[upper] - sorted_vals[lower]) * weight

    total = sum(sorted_vals)
    return {
        "count": float(count),
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
        "mean": total / count,
        "median": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
    }


def print_distribution(title: str, stats: Optional[TokenStatsConfig]) -> None:
    print(f"\n{title}:")
    if not stats:
        print("  无可用数据")
        return
    print(f"  样本数: {int(stats['count'])}")
    print(f"  最小值: {stats['min']:.0f}")
    print(f"  最大值: {stats['max']:.0f}")
    print(f"  平均值: {stats['mean']:.1f}")
    print(f"  中位数: {stats['median']:.1f}")
    print(f"  P90: {stats['p90']:.1f}")
    print(f"  P95: {stats['p95']:.1f}")
    print(f"  P99: {stats['p99']:.1f}")


def analyze_token_lengths(
    workloads: List[Any], args: argparse.Namespace, file_path: Path, mm_config: Optional[MMConfig]
) -> None:
    if not args.tokenizer or mm_config is None:
        return

    if not workloads:
        print("没有可用于 token 统计的 workloads 数据")
        return

    try:
        tokenizer, mm_config = get_tokenizer(args.tokenizer_path, mm_config)
    except Exception as exc:
        print(f"加载 tokenizer 失败: {exc}")
        return

    dataset = CustomDataset(
        workloads,
        tokenizer,
        mm_config,
        generate=True,
        group_size=args.group_size,
        max_queries=None,
    )

    if len(dataset) == 0:
        print("CustomDataset 中没有可用样本, 无法统计 token")
        return

    limit = len(dataset) if args.token_sample_limit is None else min(len(dataset), args.token_sample_limit)
    if limit == 0:
        print("Token 样本限制为 0, 跳过统计")
        return

    batch_size = min(args.token_batch_size, limit)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.token_num_workers,
        collate_fn=_passthrough_collate,
        persistent_workers=args.token_num_workers > 0,
    )

    total_lengths: List[int] = []
    prompt_lengths: List[int] = []
    response_lengths: List[int] = []
    sql_token_lengths: List[int] = []
    column_token_lengths: List[int] = []
    total_without_sql_lengths: List[int] = []
    total_without_column_lengths: List[int] = []
    total_without_special_lengths: List[int] = []
    processed = 0
    sql_token_id = getattr(mm_config, "sql_token_id", -1)
    column_token_id = getattr(mm_config, "column_token_id", -1)

    for batch in dataloader:
        for sample in batch:
            if not isinstance(sample, dict):
                continue
            input_ids = sample.get("input_ids")
            if input_ids is None:
                processed += 1
                if processed >= limit:
                    break
                continue
            total_len = int(input_ids.shape[-1])
            total_lengths.append(total_len)
            sql_tokens = int((input_ids == sql_token_id).sum().item()) if sql_token_id >= 0 else 0
            column_tokens = int((input_ids == column_token_id).sum().item()) if column_token_id >= 0 else 0
            sql_token_lengths.append(sql_tokens)
            column_token_lengths.append(column_tokens)
            total_without_sql_lengths.append(max(total_len - sql_tokens, 0))
            total_without_column_lengths.append(max(total_len - column_tokens, 0))
            total_without_special_lengths.append(max(total_len - sql_tokens - column_tokens, 0))
            labels = sample.get("labels")
            if labels is not None:
                prompt_len = int((labels == -100).sum().item())
                prompt_lengths.append(prompt_len)
                response_lengths.append(max(total_len - prompt_len, 0))

            processed += 1
            if processed >= limit:
                break
        if processed >= limit:
            break

    print("\nToken 分布 (使用 CustomDataset 生成样本):")
    print(f"  文件: {file_path}")
    print(f"  使用样本数量: {len(total_lengths)} / {len(dataset)}")
    print_distribution("总 token 数", summarize_distribution(total_lengths))
    if prompt_lengths:
        print_distribution("Prompt token 数", summarize_distribution(prompt_lengths))
    if response_lengths:
        print_distribution("响应 token 数", summarize_distribution(response_lengths))
    if sql_token_lengths:
        print_distribution("SQL token 数", summarize_distribution(sql_token_lengths))
    if column_token_lengths:
        print_distribution("列嵌入 token 数", summarize_distribution(column_token_lengths))
    if total_without_sql_lengths:
        print_distribution("剔除 SQL token 后的总 token 数", summarize_distribution(total_without_sql_lengths))
    if total_without_column_lengths:
        print_distribution("剔除列 token 后的总 token 数", summarize_distribution(total_without_column_lengths))
    if total_without_special_lengths:
        print_distribution(
            "剔除 SQL 与列 token 后的总 token 数",
            summarize_distribution(total_without_special_lengths),
        )


def analyze_workload_file(
    file_path: Path,
    data: List[Dict],
    args: argparse.Namespace,
    mm_config: Optional[MMConfig],
) -> None:
    """分析 workload 文件"""
    print(f"\n{'=' * 80}")
    print(f"分析 WORKLOAD 文件: {file_path}")
    print(f"{'=' * 80}")

    try:
        workloads = load_workloads(file_path)

        if not workloads:
            print("文件中没有 workloads 数据")
            return

        analyze_token_lengths(workloads, args, file_path, mm_config)

    except Exception as e:
        print(f"分析 workload 文件时出错: {e}")


def analyze_prompt_file(file_path: Path, data: List[Dict]) -> None:
    """分析 prompt 文件"""
    print(f"\n{'=' * 80}")
    print(f"分析 PROMPT 文件: {file_path}")
    print(f"{'=' * 80}")

    if not data:
        print("文件中没有数据")
        return

    print(f"总记录数量: {len(data)}")

    # 分析各个字段
    def analyze_field(field_name: str, extract_func) -> None:
        print(f"\n{field_name.upper()} 字段统计:")

        values = []
        for item in data:
            try:
                value = extract_func(item)
                if value is not None:
                    values.append(value)
            except Exception:
                continue

        if not values:
            print("  无有效数据")
            return

        print(f"  有效记录数: {len(values)}/{len(data)} ({len(values) / len(data) * 100:.1f}%)")

        if isinstance(values[0], str):
            # 字符串类型分析
            lengths = [len(v) for v in values]
            print(f"  长度 - 最小: {min(lengths)}, 最大: {max(lengths)}, 平均: {sum(lengths) / len(lengths):.1f}")

            # 非空字符串比例
            non_empty = sum(1 for v in values if v.strip())
            print(f"  非空字符串: {non_empty}/{len(values)} ({non_empty / len(values) * 100:.1f}%)")

        elif isinstance(values[0], list):
            # 列表类型分析
            lengths = [len(v) for v in values]
            print(f"  长度 - 最小: {min(lengths)}, 最大: {max(lengths)}, 平均: {sum(lengths) / len(lengths):.1f}")

            # 非空列表比例
            non_empty = sum(1 for v in values if v)
            print(f"  非空列表: {non_empty}/{len(values)} ({non_empty / len(values) * 100:.1f}%)")

            # 分析列表元素（如果是字符串列表）
            if lengths and all(isinstance(item, str) for sublist in values for item in sublist):
                all_items = [item for sublist in values for item in sublist]
                if all_items:
                    item_lengths = [len(item) for item in all_items]
                    print(f"  元素总数: {len(all_items)}")
                    print(
                        f"  元素长度 - 最小: {min(item_lengths)}, 最大: {max(item_lengths)}, 平均: {sum(item_lengths) / len(item_lengths):.1f}"
                    )

    # 分析各个字段
    analyze_field("input", lambda x: x.get("input"))
    analyze_field("output", lambda x: x.get("output"))
    analyze_field("instruction", lambda x: x.get("instruction"))
    analyze_field("columns", lambda x: x.get("columns"))


def analyze_file(file_path: Path, args: argparse.Namespace, mm_config: Optional[MMConfig]) -> None:
    """分析单个文件"""
    try:
        print(f"\n正在读取文件: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_type = detect_file_type(data)
        print(f"检测到文件类型: {file_type}")

        if file_type == "workload":
            analyze_workload_file(file_path, data, args, mm_config)
        elif file_type == "prompt":
            analyze_prompt_file(file_path, data)
        else:
            print("未知的文件类型，无法分析")
            print("期望的格式:")
            print("  - workload 文件: 包含 id, queries 等字段的对象数组")
            print("  - prompt 文件: 包含 input, output, instruction, columns 字段的对象数组")

    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")


def main():
    """主函数"""
    args = parse_args()
    mm_config: Optional[MMConfig]
    if args.tokenizer:
        try:
            mm_config = MMConfig.from_args(args)
        except Exception as exc:
            print(f"初始化 MMConfig 失败: {exc}")
            sys.exit(1)
    else:
        mm_config = None

    file_paths = [Path(arg) for arg in args.files]

    # 检查文件是否存在
    for file_path in file_paths:
        if not file_path.exists():
            print(f"错误: 文件不存在 {file_path}")
            sys.exit(1)
        if not file_path.is_file():
            print(f"错误: {file_path} 不是文件")
            sys.exit(1)

    print(f"将分析 {len(file_paths)} 个文件")

    # 分析每个文件
    for file_path in file_paths:
        analyze_file(file_path, args, mm_config)

    print(f"\n{'=' * 80}")
    print("分析完成!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

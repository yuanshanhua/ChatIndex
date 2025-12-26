import argparse
import sys


def parse_args_safe(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """解析 parser 的参数, 并重设 sys.argv, 使 llama-factory 的后续解析仍可正常进行"""
    args, remaining = parser.parse_known_args()
    if len(remaining) == 0:
        raise ValueError("需要提供 llama-factory 配置文件")
    elif len(remaining) > 1:
        raise ValueError(f"未知参数: {remaining}")
    sys.argv = sys.argv[0:1] + remaining
    print(f"Remaining args: {remaining}")
    return args


def compute_reward(
    basic_cost: float,
    actul_cost: float,
    *,
    total: int,
    invalid: int,
    exist: int,
    duplicate: int,
    # useless: int = 0,
) -> float:
    """
    计算奖励.

    Args:
        basic_cost: 不添加索引时的查询成本.(包括原数据库中已有的索引)
        actul_cost: 添加索引后的查询成本.
        invalid: 格式无效索引数量.
        total: 总推荐(有效)索引数量.
        exist: 已存在索引数量.
        duplicate: 重复索引数量.
        useless: 无用索引数量.
    """
    if actul_cost < 0:
        return 0
    if total == 0:
        return -1
    INVALID_PUNISHMENT = 0.2
    DUPLICATE_PUNISHMENT = 0.05
    EXIST_PUNISHMENT = 0.05
    # USELESS_PUNISHMENT = 0.1
    return 1 - actul_cost / basic_cost
    return (
        1
        - actul_cost / basic_cost
        - invalid * INVALID_PUNISHMENT
        - exist * EXIST_PUNISHMENT
        - duplicate * DUPLICATE_PUNISHMENT
        # - useless * USELESS_PUNISHMENT
    ) / total

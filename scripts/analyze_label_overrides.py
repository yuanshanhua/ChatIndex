import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from index_advisor.db import CostCache, DBOption, SizeCache, create_hypo_index_and_get_cost, get_hypo_index_size
from index_advisor.ia_logging import logger
from index_advisor.schemas import Index
from index_advisor.workload import Workload, load_workloads


log = logger.getChild("analyze_overrides")


def parse_id_list(value: str) -> set[int]:
    ids: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str), int(end_str)
            if start > end:
                start, end = end, start
            ids.update(range(start, end + 1))
        else:
            ids.add(int(part))
    return ids


def total_index_size_mb(conn, db: str, indexes: Iterable[Index], cache: SizeCache) -> float:
    if not indexes:
        return 0.0
    cursor = conn.cursor()
    try:
        total = 0.0
        for idx in indexes:
            total += get_hypo_index_size(cursor, db, idx, cache)
        return total / (1024 * 1024)
    finally:
        conn.rollback()
        cursor.close()


def get_index_size_mb(conn, db: str, idx: Index, cache: SizeCache) -> float:
    cursor = conn.cursor()
    try:
        size = get_hypo_index_size(cursor, db, idx, cache)
        return size / (1024 * 1024)
    finally:
        conn.rollback()
        cursor.close()


def order_group_with_rewards(
    conn,
    w: Workload,
    prefix: set[Index],
    group: set[Index],
    cost_cache: CostCache,
    size_cache: SizeCache,
) -> tuple[list[Index], list[float]]:
    if not group:
        return [], []

    base_cost = create_hypo_index_and_get_cost(conn, prefix, w.queries, cost_cache)
    if base_cost <= 0:
        log.warning("workload %s prefix cost 计算失败, 按原顺序返回", w.id)
        ordered = sorted(group)
        return ordered, [0.0 for _ in ordered]

    remaining = set(group)
    selected: list[Index] = []
    rewards: list[float] = []
    current_prefix = set(prefix)

    while remaining:
        best_idx: Index | None = None
        best_score = float("-inf")
        best_cost = base_cost

        for idx in remaining:
            candidate_cost = create_hypo_index_and_get_cost(conn, current_prefix | {idx}, w.queries, cost_cache)
            if candidate_cost < 0:
                continue
            size_mb = get_index_size_mb(conn, w.db, idx, size_cache)
            if size_mb <= 0 or base_cost <= 0:
                score = float("-inf")
            else:
                score = (base_cost - candidate_cost) / base_cost / size_mb
            if score > best_score:
                best_score = score
                best_idx = idx
                best_cost = candidate_cost

        if best_idx is None or best_score <= 1e-9:
            break

        selected.append(best_idx)
        rewards.append(best_score)
        current_prefix.add(best_idx)
        base_cost = best_cost
        remaining.remove(best_idx)

    if remaining:
        tail = sorted(remaining)
        selected.extend(tail)
        rewards.extend([0.0 for _ in tail])

    return selected, rewards


@dataclass
class GroupDiff:
    workload_id: int
    sqls_hash: str
    range: str
    raw_indexes: tuple[str, ...]
    override_indexes: tuple[str, ...]
    raw_rewards: tuple[float, ...]
    override_rewards: tuple[float, ...]
    prefix_cost: float
    raw_cost: float
    override_cost: float
    prefix_size: float
    raw_size: float
    override_size: float
    raw_reward: float
    override_reward: float
    raw_reward1: float
    override_reward1: float


def compute_group_diff(
    w: Workload,
    group_size: int,
    conn,
    cost_cache: CostCache,
    size_cache: SizeCache,
) -> list[GroupDiff]:
    if not w.labels or not w.label_overrides:
        return []
    if len(w.label_overrides) % group_size != 0:
        raise ValueError(f"workload {w.id} group_size={group_size} 可能不正确, 无法整除 {len(w.label_overrides)=}")

    labels: list[Index] = list(w.labels)
    results: list[GroupDiff] = []
    for gid in range(len(labels) // group_size + 1):
        start = gid * group_size
        end = min(start + group_size, len(labels))
        if start >= len(labels):
            continue
        if any(w.label_overrides.get(pos) is None for pos in range(start, end)):
            continue

        prefix = set(labels[:start])
        raw_group = set(labels[start:end])
        override_group = {w.label_overrides[pos] for pos in range(start, end)}
        if raw_group == override_group:
            continue
        raw_order, raw_rewards = order_group_with_rewards(conn, w, prefix, raw_group, cost_cache, size_cache)
        override_order, override_rewards = order_group_with_rewards(
            conn, w, prefix, override_group, cost_cache, size_cache
        )
        raw_indexes = prefix | raw_group
        overrided_indexes = prefix | override_group

        prefix_cost = create_hypo_index_and_get_cost(conn, prefix, w.queries, cost_cache)
        raw_cost = create_hypo_index_and_get_cost(conn, raw_indexes, w.queries, cost_cache)
        override_cost = create_hypo_index_and_get_cost(conn, overrided_indexes, w.queries, cost_cache)
        if prefix_cost < 0 or raw_cost < 0 or override_cost < 0:
            log.warning("workload %s group %s cost计算失败, 跳过", w.id, f"{start}-{end - 1}")
            continue

        prefix_size = total_index_size_mb(conn, w.db, prefix, size_cache)
        raw_size = total_index_size_mb(conn, w.db, raw_group, size_cache)
        override_size = total_index_size_mb(conn, w.db, override_group, size_cache)

        raw_reward = (prefix_cost - raw_cost) / raw_size if raw_size > 0 else 0.0
        raw_reward1 = raw_reward / prefix_cost if prefix_cost else 0.0

        override_reward = (prefix_cost - override_cost) / override_size if override_size > 0 else 0.0
        override_reward1 = override_reward / prefix_cost if prefix_cost else 0.0

        raw_display = tuple(str(idx) for idx in raw_order)
        override_display = tuple(str(idx) for idx in override_order)

        results.append(
            GroupDiff(
                workload_id=w.id,
                sqls_hash=w.sqls_hash,
                range=f"{start}-{end - 1}",
                raw_indexes=raw_display,
                override_indexes=override_display,
                raw_rewards=tuple(raw_rewards),
                override_rewards=tuple(override_rewards),
                prefix_cost=prefix_cost,
                raw_cost=raw_cost,
                override_cost=override_cost,
                prefix_size=prefix_size,
                raw_size=raw_size,
                override_size=override_size,
                raw_reward=raw_reward,
                override_reward=override_reward,
                raw_reward1=raw_reward1,
                override_reward1=override_reward1,
            )
        )

    return results


def format_line(diff: GroupDiff) -> str:
    raw_reward_parts = [f"{idx} (r={r:.4g})" for idx, r in zip(diff.raw_indexes, diff.raw_rewards)]
    override_reward_parts = [f"{idx} (r={r:.4g})" for idx, r in zip(diff.override_indexes, diff.override_rewards)]
    raw_reward_str = ", ".join(raw_reward_parts) or "(无变化)"
    override_reward_str = ", ".join(override_reward_parts) or "(无变化)"

    prefix_cost = diff.prefix_cost
    raw_cost = diff.raw_cost
    override_cost = diff.override_cost
    delta_raw = prefix_cost - raw_cost
    delta_override = prefix_cost - override_cost
    delta_cost_gain = delta_override - delta_raw

    prefix_size = diff.prefix_size
    raw_size = diff.raw_size
    override_size = diff.override_size

    raw_reward = diff.raw_reward
    override_reward = diff.override_reward
    delta_reward_gain = override_reward - raw_reward

    raw_reward1 = diff.raw_reward1
    override_reward1 = diff.override_reward1
    delta_reward1_gain = override_reward1 - raw_reward1

    # Multi-line output keeps related metrics together and easier to scan.
    return "\n".join(
        [
            f"workload {diff.workload_id} [{diff.range}]",
            f"  raw_index: {raw_reward_str}",
            f"  new_index: {override_reward_str}",
            f"  cost     : prefix {prefix_cost:.2f} | raw {raw_cost:.2f} (Δ{delta_raw:+.2f}) | "
            f"override {override_cost:.2f} (Δ{delta_override:+.2f}) | Δgain {delta_cost_gain:+.2f}",
            f"  size(MB) : prefix {prefix_size:.2f} | raw {raw_size:.2f} | override {override_size:.2f}",
            f"  reward   : raw {raw_reward:.4g} -> override {override_reward:.4g} (Δ{delta_reward_gain:+.4g})",
            f"  reward1  : raw {raw_reward1:.4g} -> override {override_reward1:.4g} (Δ{delta_reward1_gain:+.4g})",
        ]
    )


def filter_workloads(workloads: Sequence[Workload], ids: set[int] | None, limit: int | None) -> list[Workload]:
    res = [w for w in workloads if not ids or w.id in ids]
    if limit is not None:
        res = res[:limit]
    return res


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="分析 label_overrides 带来的成本/奖励改进", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    DBOption.add_arg_group_to(parser)
    parser.add_argument("workload_file", type=Path, help="workload JSON 路径")
    parser.add_argument("--group-size", "--gs", type=int, default=1, help="overrides 的分组大小")
    parser.add_argument("--ids", type=str, default=None, help="仅分析指定 workload id, 逗号分隔, 支持 a-b 范围")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 个匹配的 workload")

    args = parser.parse_args(argv)
    if args.group_size <= 0:
        raise ValueError("group-size 必须大于 0")

    workloads = load_workloads(args.workload_file)
    target_ids = parse_id_list(args.ids) if args.ids else None
    workloads = filter_workloads(workloads, target_ids, args.limit)
    workloads = [w for w in workloads if w.labels and w.label_overrides]
    if not workloads:
        log.info("没有包含 label_overrides 的 workload")
        return

    db_option = DBOption.from_args(args)
    connections = db_option.connections
    cost_cache = CostCache()
    size_cache = SizeCache()

    all_diffs: list[GroupDiff] = []
    for w in workloads:
        conn = connections[w.db]
        diffs = compute_group_diff(w, args.group_size, conn, cost_cache, size_cache)
        for diff in diffs:
            print(format_line(diff))
        all_diffs.extend(diffs)

    print("\n# 按 reward 降序")
    for d in sorted(all_diffs, key=lambda d: d.override_reward, reverse=True):
        print(f"reward↓ workload {d.workload_id} [{d.range}] {d.override_reward:.2f}")

    print("\n# 按 reward1 降序")
    for d in sorted(all_diffs, key=lambda d: d.override_reward1, reverse=True):
        print(f"reward1↓ workload {d.workload_id} [{d.range}] {d.override_reward1:.4g}")

    print("\n# 按 cost 下降降序")
    for d in sorted(all_diffs, key=lambda d: d.raw_cost - d.override_cost, reverse=True):
        print(f"cost↓ workload {d.workload_id} [{d.range}] {d.raw_cost - d.override_cost:.4g}")

    seen_conn: set[int] = set()
    for conn in connections.values():
        cid = id(conn)
        if cid in seen_conn:
            continue
        seen_conn.add(cid)
        conn.close()


if __name__ == "__main__":
    main()

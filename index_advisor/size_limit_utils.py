from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence, cast

from .db import DBOption, resolve_size_limits


@dataclass(frozen=True)
class SizeScenario:
    """Represents a single size-limit configuration applied across databases."""

    key: str
    label: str
    kind: Literal["size-limit", "size-factor", "unlimited"]
    value: float | None
    limits: dict[str, float | None]


@dataclass(frozen=True)
class DBLimitPlan:
    """Describes the set of limit thresholds that need to be evaluated for a single DB."""

    finite_limits: tuple[float, ...]
    include_unbounded: bool = False


def _normalize_sequence(values: Sequence[float] | None, option_name: str) -> list[float]:
    if not values:
        return []
    normalized = sorted({float(v) for v in values})
    for v in normalized:
        if v <= 0:
            raise ValueError(f"{option_name} 必须大于 0")
    return normalized


def build_size_scenarios(
    db_option: DBOption,
    size_limits: Sequence[float] | None,
    size_factors: Sequence[float] | None,
) -> list[SizeScenario]:
    """Create ordered size scenarios from CLI inputs."""

    normalized_limits = _normalize_sequence(size_limits, "--size-limit")
    normalized_factors = _normalize_sequence(size_factors, "--size-factor")

    if normalized_limits and normalized_factors:
        raise ValueError("不能同时指定 --size-limit 与 --size-factor")

    scenarios: list[SizeScenario] = []
    if normalized_limits:
        for value in normalized_limits:
            limits = resolve_size_limits(db_option, value, None)
            label = f"size-limit={value:g}MB"
            scenarios.append(
                SizeScenario(key=f"limit:{value}", label=label, kind="size-limit", value=value, limits=limits)
            )
        return scenarios

    if normalized_factors:
        for value in normalized_factors:
            limits = resolve_size_limits(db_option, None, value)
            label = f"size-factor={value:g}"
            scenarios.append(
                SizeScenario(key=f"factor:{value}", label=label, kind="size-factor", value=value, limits=limits)
            )
        return scenarios

    limits = resolve_size_limits(db_option, None, None)
    scenarios.append(
        SizeScenario(key="unlimited", label="size-limit=unlimited", kind="unlimited", value=None, limits=limits)
    )
    return scenarios


def collect_db_limit_plan(scenarios: Iterable[SizeScenario]) -> dict[str, DBLimitPlan]:
    """Aggregate per-database finite limits and whether an unbounded case is needed."""

    plan: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        for db_name, limit in scenario.limits.items():
            entry = plan.setdefault(db_name, {"finite": set(), "unbounded": False})
            if limit is None:
                entry["unbounded"] = True
            else:
                entry["finite"].add(limit)

    finalized: dict[str, DBLimitPlan] = {}
    for db_name, entry in plan.items():
        finite_limits = tuple(sorted(cast(set[float], entry["finite"])))
        include_unbounded = bool(entry["unbounded"])
        finalized[db_name] = DBLimitPlan(finite_limits=finite_limits, include_unbounded=include_unbounded)
    return finalized


__all__ = ["DBLimitPlan", "SizeScenario", "build_size_scenarios", "collect_db_limit_plan"]

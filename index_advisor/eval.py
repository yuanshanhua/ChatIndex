import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path


@dataclass
class IndexRes:
    """Evaluation result for a single index."""

    index: str
    cost_hypo: float  # 仅创建此索引的成本(使用假设索引)
    cost_real: float | None = None
    size: float | None = None  # 索引大小 (Byte)
    total_size: float = 0.0
    profit: float | None = None  # 索引带来的收益 (在一个索引序列中, 创建此索引之前的成本 - 创建此索引之后的成本)
    swirl_reward_: float | None = None

    def __post_init__(self):
        if self.swirl_reward is not None:
            self.swirl_reward_ = self.swirl_reward

    @property
    def rel_improvement(self) -> float | None:
        """计算索引的相对改进"""
        if self.profit is None or self.cost_hypo + self.profit == 0:
            return None
        return (self.profit) / (self.cost_hypo + self.profit)

    @property
    def swirl_reward(self) -> float | None:
        """swirl 使用的索引 reward 定义: 相对改进除以索引大小 (每MB的收益)"""
        if self.rel_improvement is None or self.size is None or self.size == 0:
            return None
        return self.rel_improvement / (self.size / 1024 / 1024)  # 每MB的收益


@dataclass(kw_only=True)
class Result:
    """Evaluation result for a workload."""

    workload_id: int
    db: str | None
    queries: list[str]
    sqls_hash: str
    indexes: list[IndexRes]  # 模型推荐的索引
    basic_cost_explain: float  # 使用 explain 得到的无索引时的 cost
    basic_cost_real: float | None  # 使用 explain analyze 得到的无索引时的实际查询时间
    actual_cost_hypo: float  # 使用 hypopg+explain 得到的创建假设索引后的 cost
    actual_cost_real: float | None  # 使用真实索引+explain analyze 得到的实际查询时间
    reward: float
    prompt: str
    response: str
    invalid: int
    exist: int
    duplicate: int
    useless: int

    @property
    def total_index_size(self) -> float:
        """计算所有推荐索引的总大小 (Byte)"""
        return sum(idx.size or 0.0 for idx in self.indexes)

    @property
    def swirl_reward(self) -> float:
        """计算 swirl_reward: 相对改进除以总索引大小 (每MB的收益), 此值可与 reward 字段交叉验证"""
        if self.total_index_size == 0:
            return 0.0
        return self.cost_improvement_hypo / (self.total_index_size / (1024 * 1024))

    @property
    def another_reward(self) -> float:
        """另一种 reward 计算方式: 绝对 cost 下降除以总索引大小"""
        if self.total_index_size == 0:
            return 0.0
        return (self.basic_cost_explain - self.actual_cost_hypo) / (self.total_index_size / (1024 * 1024))

    @property
    def cost_improvement_hypo(self) -> float:
        """计算假设索引的相对改进"""
        return (self.basic_cost_explain - self.actual_cost_hypo) / self.basic_cost_explain

    @property
    def cost_improvement_real(self) -> float | None:
        """计算真实索引的相对改进"""
        if self.actual_cost_real is None or self.basic_cost_real is None:
            return None
        return (self.basic_cost_real - self.actual_cost_real) / self.basic_cost_real

    @cached_property
    def workload(self) -> str:
        return "\n".join(self.queries)

    @classmethod
    def from_dict(cls, data: dict) -> "Result":
        """从字典中创建Result对象"""
        indexes = [
            IndexRes(
                index=i["index"],
                cost_hypo=i["cost_hypo"],
                cost_real=i.get("cost_real"),
                size=i.get("size"),
                profit=i.get("profit"),
            )
            for i in data["indexes"]
        ]
        return cls(
            workload_id=data.get("workload_id", 0),
            db=data.get("db"),
            queries=data["queries"],
            sqls_hash=data.get("sqls_hash", ""),
            indexes=indexes,
            basic_cost_explain=data["basic_cost_explain"],
            basic_cost_real=data.get("basic_cost_real"),
            actual_cost_hypo=data["actual_cost_hypo"],
            actual_cost_real=data.get("actual_cost_real"),
            reward=data["reward"],
            prompt=data["prompt"],
            response=data["response"],
            invalid=data["invalid"],
            exist=data["exist"],
            duplicate=data["duplicate"],
            useless=data["useless"],
        )


@dataclass
class ModelResult:
    """Evaluation results for a checkpoint."""

    name: str
    results: list[Result]
    id: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "ModelResult":
        """从字典中创建ModelResult对象"""
        results = [Result.from_dict(item) for item in data["results"]]
        name = data["name"]
        match = re.search(r"checkpoint-(\d+)", name)
        id = int(match.group(1)) if match else 0
        return cls(name, results, id)


def load_eval_results(file_path: str | Path) -> list[ModelResult]:
    """从JSON文件加载模型评估结果"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [ModelResult.from_dict(item) for item in data]

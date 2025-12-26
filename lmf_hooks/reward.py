"""
此模块用于 hook LLaMA-Factory 的 get_rewards_from_server.
它不应被其他内部模块导入.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from index_advisor import compute_reward
from index_advisor.db import DBOption, create_hypo_index_and_get_cost, get_hypo_index_size, schemas
from index_advisor.ia_logging import logger
from index_advisor.msg import get_continue_count, get_workload_id
from index_advisor.schemas import Index, parse_indexes_str
from index_advisor.workload import Workload


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logger.getChild("reward")


@dataclass
class State:
    db: DBOption
    workloads_by_id: dict[int, Workload]
    reward: Literal["format", "cost", "div"]
    group_size: int = 1

    def __post_init__(self):
        if self.reward == "format":
            logger.warning("当前 reward 模式设置为 [format], 仅考虑格式正确, 不考虑索引效果, 请确保此行为符合预期.")
        if self.group_size <= 0:
            raise ValueError("group_size 必须大于 0")


state: State = None  # type: ignore[assignment]


def rl_init(
    db: DBOption,
    workloads: list[Workload],
    reward: Literal["format", "cost", "div"] = "cost",
    *,
    group_size: int = 1,
):
    """
    初始化 RL 训练环境

    Args:
        db (DBOption): 数据库连接选项
        tmpl (template): prompt 模板
        workloads (list[Workload]): 工作负载元信息
        reward: 奖励计算方式. format 仅考虑格式正确, cost 同时考虑索引效果, div 除以推荐索引数量.
    """
    global state
    # 验证所有 workload 的 db 均在 DBOption 中
    dbs = {w.db for w in workloads}
    assert all(name in schemas and name in db.connections for name in dbs), (
        "所有 workload 所属的 db 必须存在于 schemas 且在 DBOption 中具有连接."
    )
    state = State(db=db, workloads_by_id={w.id: w for w in workloads}, reward=reward, group_size=group_size)


def get_reward(msg: tuple[str, str]) -> float:
    prompt, resp = msg
    logger.debug(f"Msg: {msg}")
    logger.info("开始处理")

    workload_id = get_workload_id(prompt)
    logger.info(f"Workload ID: {workload_id}")
    workload = state.workloads_by_id[workload_id]
    logger.info(f"Workload SQLs Hash: {workload.sqls_hash}")
    logger.info(f"查询数量: {len(workload.sqls)}")
    db_name = workload.db
    logger.info(f"database: {db_name}")

    basic_cost = create_hypo_index_and_get_cost(state.db.connections[db_name], [], workload.queries)
    logger.info(f"基准: {basic_cost:.2f}")

    indexes, (invalid, exist, duplicate) = parse_indexes_str(resp, schemas[db_name], True)

    if True:  # todo 使用参数控制
        return _get_one_turn_reward(prompt, indexes, workload, basic_cost)  # 多轮对话模式

    total = len(indexes)
    logger.info(f"推荐索引: {indexes}")
    logger.info(f"过滤索引: 无效 {invalid} 已存在 {exist} 重复 {duplicate}")
    if state.reward == "format":
        # 仅考虑格式正确, 不考虑索引效果
        actul_cost = 0
        total = 1  # 避免推荐减少
    else:
        actul_cost = create_hypo_index_and_get_cost(state.db.connections[db_name], indexes, workload.queries)
        if state.reward != "div":
            total = 1
    logger.info(f"Cost: {actul_cost}")

    reward = compute_reward(
        basic_cost,
        actul_cost,
        total=total,
        invalid=invalid,
        exist=exist,
        duplicate=duplicate,
    )
    logger.info(f"Reward: {reward}")
    return reward


def _get_one_turn_reward(prompt: str, indexes: list[Index], w: Workload, basic_cost: float) -> float:
    logger.info(f"本轮推荐索引: {indexes}")
    expected = max(1, state.group_size)
    if len(indexes) < expected:
        logger.warning(f"推荐索引数量 {len(indexes)} 少于 {expected}")
        logger.info("Reward: 0")
        return 0.0

    if len(indexes) > expected:
        logger.warning(f"仅计算前 {expected} 个索引的奖励")
    selected = indexes[:expected]

    # 需要获取之前的索引, 即构造的多轮对话中前几轮的索引
    # 因为这些索引实际来自于 workload 的原始 labels, 因此只需要计算对话有几轮, 然后取对应的 labels 即可
    count = get_continue_count(prompt)
    # 每个 continue 指令前都有 group_size 个来自 labels 的索引, 因此 count 即为前面索引的数量
    prev_total = count * expected
    prev_indexes = w.labels[:prev_total] if w.labels else []
    logger.debug(f"之前的索引: {prev_indexes}")
    conn = state.db.connections[w.db]
    prev_cost = create_hypo_index_and_get_cost(conn, prev_indexes, w.queries)
    logger.debug(f"之前索引的 Cost: {basic_cost:.2f} Profit: {basic_cost - prev_cost:.2f}")
    cur_cost = create_hypo_index_and_get_cost(conn, prev_indexes + selected, w.queries)
    size = 0.0
    with conn.cursor() as cursor:
        for index in selected:
            size += get_hypo_index_size(cursor, w.db, index) / (1024 * 1024)
    logger.info(f"Cost: {cur_cost:.2f}")
    logger.info(f"本轮索引 profit: {prev_cost - cur_cost:.2f}")
    logger.info(f"索引总大小: {size:.2f}MB")
    reward = compute_reward(
        prev_cost,
        cur_cost,
        total=1,
        invalid=0,
        exist=0,
        duplicate=0,
    )
    reward /= size / 1024
    logger.info(f"Reward: {reward}")
    return reward


# not used
def get_token_level_reward(
    prompts: list["torch.Tensor"],
    responses: list["torch.Tensor"],
    tokenizer: "PreTrainedTokenizer",
) -> list["torch.Tensor"]:
    s_prompts = tokenizer.batch_decode(prompts, skip_special_tokens=True)
    return [_get_token_level_reward(prompt, response, tokenizer) for prompt, response in zip(s_prompts, responses)]


def _get_token_level_reward(
    prompt: str,
    response: "torch.Tensor",
    tokenizer: "PreTrainedTokenizer",
) -> "torch.Tensor":
    # 完整的 response
    resp = tokenizer.decode(response.cpu(), skip_special_tokens=True)

    logger.debug(f"Msg: {(prompt, resp)}")
    logger.info("开始处理")

    workload_id = get_workload_id(prompt)
    logger.info(f"Workload ID: {workload_id}")

    workload = state.workloads_by_id[workload_id]
    logger.info(f"Workload SQLs Hash: {workload.sqls_hash}")
    logger.info(f"查询数量: {len(workload.sqls)}")

    db_name = workload.db
    logger.info(f"database: {db_name}")
    conn = state.db.connections[db_name]

    basic_cost = create_hypo_index_and_get_cost(conn, [], workload.queries)
    logger.info(f"基准: {basic_cost}")

    prefix_ids = []
    line_idx = [-1]  # 换行符/终结符在响应字符串中的索引. line_idx[i]+1 到 line_idx[i+1] 是第 i 行的内容
    prefix_idx = set()  # 当前索引集合
    prefix_cost = basic_cost  # 当前索引的 cost
    rewards = torch.zeros_like(response, dtype=torch.float)
    for i, token in enumerate(response):
        prefix_ids.append(token.item())
        prefix = tokenizer.decode(prefix_ids, skip_special_tokens=True)
        if ((r := prefix.rfind("\n")) > line_idx[-1]) or prefix == resp:  # 发现新行或结束
            line_idx.append(r)
            line = prefix[line_idx[-2] + 1 : line_idx[-1]]
            if prefix == resp:  # 处理最后一行
                line = prefix[line_idx[-2] + 1 :]
            logger.debug(f"处理第 {len(line_idx) - 1} 行: {line}")
            indexes, (invalid, exist, duplicate) = parse_indexes_str(line, schemas[db_name])
            logger.info(f"本行索引: {indexes}")
            prefix_idx |= set(indexes)
            idx_cost = create_hypo_index_and_get_cost(conn, prefix_idx, workload.queries)
            logger.info(f"Cost: {idx_cost}")
            logger.info(f"Line Profit: {prefix_cost - idx_cost}")  # 本行索引带来的收益
            line_reward = compute_reward(
                prefix_cost,
                idx_cost,
                total=len(indexes) if state.reward == "div" else 1,
                invalid=invalid,
                exist=exist,
                duplicate=duplicate,
            )
            rewards[i] = line_reward
            logger.info(f"Line Reward: {line_reward}")
            prefix_cost = idx_cost
            if prefix == resp:  # 已处理所有响应 token
                return rewards
    return rewards

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import matplotlib.pyplot as plt
from PIL import Image


# 日志字段类型定义, 便于类型检查
class LogRecord(TypedDict, total=False):
    type: str
    current_steps: int
    loss: float
    avg_loss: float
    eval_loss: float
    reward: float
    eval_reward: float
    ppo_buffer_size: int


def load_jsonl(path: Path) -> list[LogRecord]:
    records: list[LogRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj: LogRecord = json.loads(line)
            except json.JSONDecodeError:
                # 跳过格式异常的行, 保证后续绘图不中断
                continue
            records.append(obj)
    return records


def detect_log_type(records: list[LogRecord]) -> Literal["rl", "sft"]:
    """简单启发式: 包含 reward 或 PPO 相关字段视为 RL, 否则视为 SFT"""
    for r in records:
        if "reward" in r:
            return "rl"
        if r.get("ppo_buffer_size") is not None:
            return "rl"
    return "sft"


@dataclass
class SFTMetrics:
    train_steps: list[int]
    train_loss: list[float]
    train_avg_loss: list[float]
    eval_steps: list[int]
    eval_loss: list[float]


@dataclass
class RLMetrics:
    train_steps: list[int]
    train_loss: list[float]
    train_reward: list[float]
    eval_steps: list[int]
    eval_loss: list[float]
    eval_reward: list[float]


def collect_metrics_sft(records: list[LogRecord]) -> SFTMetrics:
    train_steps: list[int] = []
    train_loss: list[float] = []
    train_avg_loss: list[float] = []

    eval_steps: list[int] = []
    eval_loss: list[float] = []

    # 使用全局单调步数, 避免 resumed 造成的回退
    step_counter = 0
    last_step_value: int | None = None

    for r in records:
        typ = r.get("type")
        current_steps = r.get("current_steps")

        # 将 current_steps 标准化为单调步数
        if isinstance(current_steps, int):
            if last_step_value is None:
                step_counter = current_steps
            elif current_steps >= last_step_value:
                step_counter += current_steps - last_step_value
            else:
                step_counter += 1
            last_step_value = current_steps
        else:
            step_counter += 1

        if typ == "train":
            loss = r.get("loss")
            avg_loss = r.get("avg_loss")
            if loss is not None:
                train_steps.append(step_counter)
                train_loss.append(loss)
            if avg_loss is not None:
                # 与 train_steps 对齐, 后续切片时保持一致
                train_avg_loss.append(avg_loss)
        elif typ == "eval":
            eloss = r.get("eval_loss")
            if eloss is not None:
                eval_steps.append(step_counter)
                eval_loss.append(eloss)

    return SFTMetrics(
        train_steps=train_steps,
        train_loss=train_loss,
        train_avg_loss=train_avg_loss,
        eval_steps=eval_steps,
        eval_loss=eval_loss,
    )


def collect_metrics_rl(records: list[LogRecord]) -> RLMetrics:
    train_steps: list[int] = []
    train_loss: list[float] = []
    train_reward: list[float] = []

    eval_steps: list[int] = []
    eval_loss: list[float] = []
    eval_reward: list[float] = []

    step_counter = 0
    last_step_value: int | None = None

    for r in records:
        current_steps = r.get("current_steps")

        if isinstance(current_steps, int):
            if last_step_value is None:
                step_counter = current_steps
            elif current_steps >= last_step_value:
                step_counter += current_steps - last_step_value
            else:
                step_counter += 1
            last_step_value = current_steps
        else:
            step_counter += 1

        # RL 日志可能缺少明确 type, 默认视为训练
        typ = r.get("type")
        if typ == "eval":
            eloss = r.get("eval_loss")
            ereward = r.get("eval_reward")
            if eloss is not None:
                eval_steps.append(step_counter)
                eval_loss.append(eloss)
            if ereward is not None:
                eval_reward.append(ereward)
        else:
            loss = r.get("loss")
            reward = r.get("reward")
            if loss is not None:
                train_steps.append(step_counter)
                train_loss.append(loss)
            if reward is not None:
                train_reward.append(reward)

    if eval_reward and all(x == 0 for x in eval_reward):
        # 清理无信息的奖励列
        eval_reward = []
        eval_steps = [] if not eval_loss else eval_steps

    return RLMetrics(
        train_steps=train_steps,
        train_loss=train_loss,
        train_reward=train_reward,
        eval_steps=eval_steps,
        eval_loss=eval_loss,
        eval_reward=eval_reward,
    )


def plot_lines(xy_list: list[tuple[list[int], list[float], str]], title: str) -> Image.Image:
    plt.figure(figsize=(8, 4), dpi=160)
    for x, y, label in xy_list:
        if not x or not y:
            continue
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel("steps")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 将图像暂存到临时文件, 再用 PIL 读取便于后续拼接
    tmp_path = Path.cwd() / f".__plot_tmp_{os.getpid()}_{abs(hash(title))}.png"
    plt.tight_layout()
    plt.savefig(tmp_path)
    plt.close()
    img = Image.open(tmp_path)
    # 立刻删除临时文件, 避免残留
    try:
        tmp_path.unlink(missing_ok=True)
    except OSError:
        pass
    return img


def compose_vertical(images: list[Image.Image]) -> Image.Image | None:
    images = [img for img in images if img is not None]
    if not images:
        return None

    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)
    canvas = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))
    y = 0
    for img in images:
        canvas.paste(img, (0, y))
        y += img.size[1]
    return canvas


def main():
    parser = argparse.ArgumentParser(description="读取 trainer 日志 (jsonl) 并输出指标图像")
    parser.add_argument("log_files", nargs="+", type=Path, help="trainer 日志文件路径 (jsonl 格式)")
    parser.add_argument("--name", "-n", help="输出图片文件名 (默认: metrics.png)", default="metrics.png")
    args = parser.parse_args()

    for log_path in args.log_files:
        if not log_path.exists():
            print(f"日志文件不存在: {log_path}")
            continue
        print(f"处理日志文件: {log_path}")
        process_log_file(log_path, args.name)


def process_log_file(log_path: Path, name: str):
    records = load_jsonl(log_path)
    if not records:
        print("未找到有效日志记录, 无法绘图")
        return

    log_type = detect_log_type(records)

    plot_images: list[Image.Image] = []
    if log_type == "sft":
        m = collect_metrics_sft(records)
        # 损失面板: 训练 loss / 平滑 loss / 验证 loss
        loss_lines: list[tuple[list[int], list[float], str]] = []
        step_loss: list[tuple[list[int], list[float], str]] = []
        if m.train_steps and m.train_loss:
            step_loss.append((m.train_steps, m.train_loss, "step-loss"))
        if m.train_avg_loss:
            loss_lines.append((m.train_steps[: len(m.train_avg_loss)], m.train_avg_loss, "avg-train-loss"))
        if m.eval_steps and m.eval_loss:
            loss_lines.append((m.eval_steps, m.eval_loss, "eval-loss"))
        if loss_lines:
            plot_images.append(plot_lines(loss_lines, title="SFT Loss"))
        if step_loss:
            plot_images.append(plot_lines(step_loss, title="SFT Step Loss"))
    else:
        m = collect_metrics_rl(records)
        loss_lines: list[tuple[list[int], list[float], str]] = []
        if m.train_steps and m.train_loss:
            loss_lines.append((m.train_steps, m.train_loss, "train-loss"))
        if m.eval_steps and m.eval_loss:
            loss_lines.append((m.eval_steps, m.eval_loss, "eval-loss"))
        if loss_lines:
            plot_images.append(plot_lines(loss_lines, title="RL Losses"))

        reward_lines: list[tuple[list[int], list[float], str]] = []
        if m.train_steps and m.train_reward:
            reward_lines.append((m.train_steps[: len(m.train_reward)], m.train_reward, "train-reward"))
        if m.eval_steps and m.eval_reward:
            reward_lines.append((m.eval_steps[: len(m.eval_reward)], m.eval_reward, "eval-reward"))
        if reward_lines:
            plot_images.append(plot_lines(reward_lines, title="RL Rewards"))

    combined = compose_vertical(plot_images)
    if combined is None:
        print("日志中没有可绘制的指标")
        return

    out_path = log_path.parent / name
    combined.save(out_path)
    print(f"指标图已保存: {out_path}")


if __name__ == "__main__":
    main()

import argparse
import os
import sys
from pathlib import Path

import torch.distributed as dist
import yaml
from transformers.trainer_callback import TrainerCallback
from trl import PPOConfig

from index_advisor.db import DBOption
from index_advisor.ia_logging import log_to_file, logger
from index_advisor.mm.model import MMConfig
from index_advisor.utils import resolve_ckpt
from index_advisor.workload import load_workloads
from lmf_hooks.model import mm_init
from lmf_hooks.reward import rl_init


logger = logger.getChild("ppo")


def run_train():
    parser = argparse.ArgumentParser(
        description="Train using LLaMA-Factory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("workload_file", help="workload 文件", type=str, default="data/1.workload.json")
    parser.add_argument("config_file", help="llama-factory 配置文件", type=str, default="config/train-qwen-mm.yaml")
    parser.add_argument(
        "--adapter-dir", help="lora adapter path, 将覆盖配置文件中的 adapter_name_or_path", type=str, default=None
    )
    parser.add_argument("--gen-batch-size", help="PPO rollout batchsize", type=int, default=None)
    parser.add_argument("--output-dir", help="输出目录, 将覆盖配置文件中的 output_dir", type=str, default="")
    parser.add_argument("--eval-data-file", help="评估 workload 文件路径（可选）", type=str, default=None)
    parser.add_argument("--eval-batch-size", help="评估 batchsize", type=int, default=16)
    parser.add_argument(
        "--eval-set-percent",
        help="评估集比例 (0-1), 将从 sft 数据集中取出一定比例的数据供评估使用",
        type=float,
        default=0.0,
    )
    parser.add_argument("--size-limit", "-s", help="单个 workload 推荐索引大小上限 (MB)", type=int, default=500)
    parser.add_argument("--count-limit", "-n", help="单个 workload 推荐索引数量上限", type=int, default=5)
    parser.add_argument("--group-size", "--gs", help="每轮推荐的索引数量", type=int, default=1)
    parser.add_argument("--multi-gen", help="启用多轮生成并为每一轮给出 token-level reward", action="store_true")
    parser.add_argument("--desc", help="训练任务的描述", type=str, default="训练模型")
    parser.add_argument("--debug", help="调试模式", action="store_true")
    parser.add_argument("--reward", help="奖励计算方式", type=str, default="cost", choices=["format", "cost", "div"])
    DBOption.add_arg_group_to(parser)
    MMConfig.add_arg_group_to(parser)
    # PPO 相关参数
    ppoargs = parser.add_argument_group("PPO 相关", "调整 PPO 训练的参数")
    ppoargs.add_argument(
        "--ppo-adap-kl-ctrl",
        "--akl",
        dest="ppo_adap_kl_ctrl",
        help="启用自适应 KL 控制 (adap_kl_ctrl), 默认启用",
        action="store_true",
    )
    ppoargs.add_argument(
        "--ppo-no-adap-kl-ctrl",
        "--lkl",
        dest="ppo_adap_kl_ctrl",
        help="禁用自适应 KL 控制 (adap_kl_ctrl), 使用线性控制",
        action="store_false",
    )
    ppoargs.set_defaults(ppo_adap_kl_ctrl=True)
    ppoargs.add_argument(
        "--ppo-init-kl-coef",
        "--klc",
        help="初始 KL 系数 (init_kl_coef)，默认 0.2",
        type=float,
        default=0.2,
    )
    ppoargs.add_argument(
        "--ppo-kl-penalty",
        "--klp",
        help="KL 惩罚类型 (kl_penalty) 'kl': model_logp - ref_logp,  'abs': abs(kl),  'mse': mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution",
        type=str,
        choices=["kl", "abs", "mse", "full"],
        default="kl",
    )
    ppoargs.add_argument(
        "--ppo-target",
        "--klt",
        help="adaptive KL 控制的目标值 (target)，默认 6",
        type=float,
        default=None,
    )
    ppoargs.add_argument(
        "--ppo-lam",
        help="GAE λ (lam)，默认 0.95",
        type=float,
        default=None,
    )
    ppoargs.add_argument(
        "--ppo-gamma",
        help="GAE 折扣因子 γ (gamma)，默认 1.0",
        type=float,
        default=None,
    )
    ppoargs.add_argument(
        "--ppo-cliprange",
        help="PPO 策略裁剪范围 (cliprange)，默认 0.2",
        type=float,
        default=None,
    )
    ppoargs.add_argument(
        "--ppo-cliprange-value",
        help="价值头裁剪范围 (cliprange_value)，默认 0.2",
        type=float,
        default=None,
    )
    ppoargs.add_argument(
        "--ppo-vf-coef",
        help="价值损失系数 (vf_coef)，默认 0.1",
        type=float,
        default=None,
    )
    ppoargs.add_argument(
        "--ppo-horizon",
        help="adaptive KL 控制的时间跨度 (horizon)，默认 10000",
        type=float,
        default=None,
    )
    ppoargs.add_argument(
        "--ppo-early-stopping",
        help="KL 过高时是否提前停止 PPO 优化 (early_stopping)，默认关闭",
        action="store_true",
        default=None,
    )
    ppoargs.add_argument(
        "--ppo-target-kl",
        help="early_stopping 的目标 KL 值 (target_kl)，默认 1；当 KL 超过该值的 1.5 倍时提前停止",
        type=float,
        default=None,
    )
    ppoargs.add_argument(
        "--ppo-ratio-threshold",
        help="PPO 损失中 ratio 的阈值 (ratio_threshold)，超过该阈值的样本将被跳过，默认 10",
        type=float,
        default=None,
    )

    all_args = sys.argv  # for logging
    args = parser.parse_args()

    if args.eval_set_percent < 0 or args.eval_set_percent >= 1:
        raise ValueError("--eval-set-percent 必须在 [0, 1) 范围内")
    if args.eval_set_percent > 0 and (args.eval_data_file is not None):
        raise ValueError("不能同时指定 --eval-data-file 和 --eval-set-percent")
    if args.group_size <= 0:
        raise ValueError("--group-size 必须大于 0")

    mm_config = MMConfig.from_args(args)

    # 多卡训练
    # 获取当前进程的 rank
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    global_rank = int(os.environ.get("RANK", -1))

    # 获取总进程数
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"Local Rank: {local_rank}, Global Rank: {global_rank}, World Size: {world_size}")
    # 主进程或单卡训练打印日志
    if local_rank in [-1, 0]:
        log_to_file("train", "DEBUG" if args.debug else "INFO", args.desc, all_args, args.output_dir)

    mm_init(mm_config)
    if not os.path.exists(args.workload_file):
        raise ValueError(f"workload 文件 {args.workload_file} 不存在")
    if not os.path.isfile(args.workload_file):
        raise ValueError(f"workload 文件 {args.workload_file} 不是一个文件")
    print(f"Loading workload from {args.workload_file}")
    db_opt = DBOption.from_args(args)
    rl_init(db_opt, load_workloads(args.workload_file), args.reward, group_size=args.group_size)

    from llamafactory.hparams import get_train_args
    from llamafactory.train.callbacks import LogCallback, ReporterCallback
    from llamafactory.train.ppo import run_ppo

    assert args.config_file.endswith((".yaml", ".yml")), "配置文件必须是 yaml 格式"

    lmf_args = yaml.safe_load(Path(args.config_file).absolute().read_text())

    callbacks: list[TrainerCallback] = []
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(lmf_args)
    if args.adapter_dir:
        args.adapter_dir = resolve_ckpt(args.adapter_dir)
        model_args.adapter_name_or_path = [args.adapter_dir]  # type: ignore
        if not os.path.exists(args.adapter_dir):
            raise ValueError(f"指定的 adapter_dir 不存在: {args.adapter_dir}")
    elif model_args.adapter_name_or_path:
        raise ValueError("请通过 --adapter-dir 指定 LoRA adapter 路径, 配置文件中的 adapter_name_or_path 将被忽略")

    if args.output_dir:
        training_args.output_dir = args.output_dir
        logger.info(f"使用命令行指定的输出目录: {training_args.output_dir}, 将覆盖配置文件中的设置")

    callbacks.append(LogCallback())
    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))  # add to last

    assert finetuning_args.stage == "ppo", "此脚本仅用于 PPO 训练"

    # 打印将要使用的 PPO 关键参数（命令行与 finetuning_args 对比）
    default_config = PPOConfig()

    if args.ppo_target is not None and args.ppo_target != finetuning_args.ppo_target:
        logger.info(f"  提示: 命令行参数覆盖配置文件 ppo_target: {finetuning_args.ppo_target} -> {args.ppo_target}")

    logger.info("PPO 训练参数:")
    logger.info(f"  使用 adaptive KL 控制: {args.ppo_adap_kl_ctrl}")
    logger.info(f"  初始 KL 系数 (init_kl_coef): {args.ppo_init_kl_coef or default_config.init_kl_coef}")
    logger.info(f"  KL 惩罚类型 (kl_penalty): {args.ppo_kl_penalty or default_config.kl_penalty}")
    if args.ppo_adap_kl_ctrl:
        logger.info(f"  adaptive KL 控制的目标值 (target): {args.ppo_target or default_config.target}")
        logger.info(f"  adaptive KL 控制的时间跨度 (horizon): {args.ppo_horizon or default_config.horizon}")
    logger.info(f"  GAE λ (lam): {args.ppo_lam or default_config.lam}")
    logger.info(f"  折扣因子 γ (gamma): {args.ppo_gamma or default_config.gamma}")
    logger.info(f"  策略裁剪范围 (cliprange): {args.ppo_cliprange or default_config.cliprange}")
    logger.info(f"  价值裁剪范围 (cliprange_value): {args.ppo_cliprange_value or default_config.cliprange_value}")
    logger.info(f"  价值损失系数 (vf_coef): {args.ppo_vf_coef or default_config.vf_coef}")
    logger.info(f"  early_stopping: {args.ppo_early_stopping or default_config.early_stopping}")
    if args.ppo_early_stopping:
        logger.info(f"  early_stopping 目标 KL 值 (target_kl): {args.ppo_target_kl or default_config.target_kl}")

    run_ppo(
        model_args,
        data_args,
        training_args,
        finetuning_args,
        generating_args,
        callbacks,
        workload_file=args.workload_file,
        eval_data_file=args.eval_data_file,
        eval_set_percent=args.eval_set_percent,
        eval_batch_size=args.eval_batch_size,
        eval_size_limit=args.size_limit * 1024 * 1024,
        eval_count_limit=args.count_limit,
        gen_batch_size=args.gen_batch_size,
        token_level_reward_fn=None,
        db_option=db_opt,
        group_size=args.group_size,
        multi_gen=args.multi_gen,
        ppo_adap_kl_ctrl=args.ppo_adap_kl_ctrl,
        ppo_init_kl_coef=args.ppo_init_kl_coef,
        ppo_kl_penalty=args.ppo_kl_penalty,
        ppo_target=args.ppo_target,
        ppo_horizon=args.ppo_horizon,
        ppo_early_stopping=args.ppo_early_stopping,
        ppo_target_kl=args.ppo_target_kl,
        ppo_ratio_threshold=args.ppo_ratio_threshold,
        ppo_lam=args.ppo_lam,
        ppo_gamma=args.ppo_gamma,
        ppo_cliprange=args.ppo_cliprange,
        ppo_cliprange_value=args.ppo_cliprange_value,
        ppo_vf_coef=args.ppo_vf_coef,
    )

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Failed to destroy process group: {e}.")


if __name__ == "__main__":
    run_train()

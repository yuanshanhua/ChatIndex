import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from index_advisor import parse_args_safe
from index_advisor.db import DBOption
from index_advisor.evaluator import ParallelEvaluator
from index_advisor.ia_logging import log_to_file, logger
from index_advisor.mm.model import MMConfig
from index_advisor.utils import parse_ranges
from lmf_hooks.model import mm_init


try:
    # 适配 Ascend npu, 不要删除
    from torch_npu.contrib import transfer_to_npu  # type: ignore
except ImportError:
    pass

logger = logger.getChild("eval")


def run_eval():
    parser = argparse.ArgumentParser(
        description="Evaluate models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    DBOption.add_arg_group_to(parser)
    MMConfig.add_arg_group_to(parser)
    parser.add_argument("workload_file", help="用于测试的 workload 文件")
    parser.add_argument("output_file", help="测试结果输出路径")
    parser.add_argument("--round", "-n", help="推理轮次. 理想情况下总推荐数 = round*group_size", type=int, default=10)
    parser.add_argument("--group-size", "--gs", help="每轮推荐的索引数量", type=int, default=1)
    parser.add_argument(
        "--eval-set-percent",
        help="⚠️ 注意, 此参数含义与训练脚本保持一致. 如设为 0.1 则表示使用 90% 的数据进行评估.",
        type=float,
        default=0.0,
    )
    parser.add_argument("--gpu", help="要使用的 GPU. 格式同 ckpt", default="0")
    parser.add_argument("-b", "--batch-size", help="批量推理 batch size", type=int, default=32)
    parser.add_argument("--adapter-dir", help="adapter path, 将覆盖配置文件中的 adapter_name_or_path", default=None)
    parser.add_argument("--desc", help="评估任务的描述", default="评估模型")
    parser.add_argument(
        "--ckpt",
        help="需评估的 checkpoint. 以逗号分隔, 连续多个可使用 start:end:step 表示, 包含 end. 例如 100:300:100,500,700:800:50 代表 100,200,300,500,700,750,800. "
        "此参数将评估 --adapter-dir 下所有 checkpoint-* 中的模型."
        "当启用多模态但未设置 --mm*-pckpt 时会加载对应目录中的 projector checkpoint.",
        default="all",
    )
    parser.add_argument("-r", "--real", help="创建真实索引并实际执行查询", action="store_true")
    parser.add_argument("--ptype", choices=["general", "prompt1", "plus"], default="general", help="指定 prompt 变体")
    parser.add_argument("--debug", help="调试模式", action="store_true")

    all_args = sys.argv  # for logging
    args = parse_args_safe(parser)
    log_to_file("eval", "DEBUG" if args.debug else "INFO", args.desc, all_args)

    # check args
    if args.eval_set_percent < 0 or args.eval_set_percent >= 1:
        raise ValueError("--eval-set-percent 必须在 [0, 1) 范围内")
    batch_size = args.batch_size
    if batch_size <= 0:
        raise ValueError("batch size 必须大于 0")
    if args.group_size <= 0:
        raise ValueError("group size 必须大于 0")

    # 仅用于检查相关 args 是否有效, 实际由各进程从 args 中创建
    mm_config = MMConfig.from_args(args)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # parse checkpoints
    if args.adapter_dir is None:
        ckpts = None
    elif args.ckpt == "all":
        ckpts = Path(args.adapter_dir).parent.glob("checkpoint-*")
    elif not args.ckpt:
        raise ValueError("指定 --adapter-dir 同时必须指定 --ckpt")
    else:
        ckpts = [Path(args.adapter_dir).parent / f"checkpoint-{s}" for s in parse_ranges(args.ckpt)]
        ckpts = [p for p in ckpts if p.is_dir()]
    logger.info(f"使用检查点: {ckpts}")

    # parse devices
    device_ids = parse_ranges(args.gpu, True)  # 允许指定同一 GPU 多次
    logger.info(f"使用 gpu: {device_ids}")

    # init mm
    mm_init(mm_config)

    db_option = DBOption.from_args(args) if args.db is not None else None

    evaluator = ParallelEvaluator(
        args=args,
        db_option=db_option,
        workload_file=args.workload_file,
        round=args.round,
        ckpt_paths=ckpts,
        real=args.real,
        device_ids=device_ids,
        group_size=args.group_size,
    )

    # 评估所有检查点
    all_res = evaluator.evaluate(batch_size)

    # save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_res], f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    run_eval()

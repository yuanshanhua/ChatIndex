import os
import subprocess
import sys
from argparse import ArgumentParser

from index_advisor.device import set_device
from llamafactory.extras.misc import find_available_port
from scripts import sft_projector


def main() -> None:
    parser = ArgumentParser(description="多卡 SFT 训练脚本")
    parser.add_argument("--gpu", default=None, help="指定使用的 GPU. 将覆盖环境变量设置.")
    args, remaining = parser.parse_known_args()
    device_ids = set_device(args.gpu)
    print(f"使用 GPU: {device_ids}")

    # 获取分布式训练参数
    nnodes = os.getenv("NNODES", "1")
    node_rank = os.getenv("NODE_RANK", "0")
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", str(find_available_port()))

    print(f"初始化 {len(device_ids)} 个分布式任务，地址: {master_addr}:{master_port}")
    if int(nnodes) > 1:
        print(f"多节点训练启用: 节点数: {nnodes}, 节点rank: {node_rank}")

    cmd = [
        "torchrun",
        "--nnodes",
        nnodes,
        "--node_rank",
        node_rank,
        "--nproc_per_node",
        "gpu",
        "--master_addr",
        master_addr,
        "--master_port",
        master_port,
        sft_projector.__file__,
    ] + remaining

    print(f"执行命令: {' '.join(cmd)}")

    process = subprocess.run(cmd, env=os.environ)
    sys.exit(process.returncode)


if __name__ == "__main__":
    main()

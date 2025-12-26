import os
import subprocess
import sys
from argparse import ArgumentParser

from index_advisor.device import set_device
from llamafactory.extras.misc import find_available_port
from scripts import train


def main():
    parser = ArgumentParser(description="多卡 RL 训练脚本")
    parser.add_argument("--gpu", default=None, help="指定使用的 GPU. 将覆盖环境变量设置.")
    args, remaining = parser.parse_known_args()
    device_ids = set_device(args.gpu)
    print(f"使用 GPU: {device_ids}")

    nnodes = os.getenv("NNODES", "1")
    node_rank = os.getenv("NODE_RANK", "0")
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", str(find_available_port()))
    if int(nnodes) > 1:
        print(f"Multi-node training enabled: num nodes: {nnodes}, node rank: {node_rank}")

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
        train.__file__,
    ] + remaining

    print(f"执行命令: {' '.join(cmd)}")

    process = subprocess.run(cmd, env=os.environ)
    sys.exit(process.returncode)


if __name__ == "__main__":
    main()

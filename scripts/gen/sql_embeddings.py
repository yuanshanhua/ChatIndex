import argparse
import hashlib
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path
from typing import cast

import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer

from index_advisor.dataset import read_file
from index_advisor.device import set_env_devices


try:
    # 适配 Ascend npu, 不要删除
    from torch_npu.contrib import transfer_to_npu  # type: ignore
except ImportError:
    pass


def generate_embeddings(
    queue: "mp.SimpleQueue[dict[str, object]]",
    device: int,
    queries: list[str],
    model_path: str,
    batch_size: int,
):
    """
    使用指定模型为 SQL 查询生成嵌入.

    Args:
        queue: 结果返回队列
        device: 推理使用的设备 ID
        queries: SQL 查询字符串列表
        model_path: 模型目录路径
        batch_size: 每批处理的查询数量

    Returns:
        将查询字符串映射到其嵌入的字典
    """
    print(f"GPU {device} 加载模型: {model_path}")

    # 加载分词器和模型
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to("cuda")
    model.eval()

    # 确保存在填充标记
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embeddings: dict[str, object] = {}

    print(f"GPU {device} {batch_size=} {len(queries)=}")

    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i : i + batch_size]
            print(
                f"GPU {device} 处理批次 {i // batch_size + 1}/{(len(queries) + batch_size - 1) // batch_size} "
                f"(查询 {i + 1}-{min(i + batch_size, len(queries))})"
            )

            # 对查询批次进行分词
            inputs = tokenizer.__call__(
                batch_queries,
                return_tensors="pt",
                padding=True,
                padding_side="left",
            ).to("cuda")
            inputs = cast("dict[str,torch.Tensor]", inputs)
            assert inputs["input_ids"].shape[0] == len(batch_queries)

            # 获取模型输出
            outputs = model(**inputs)

            # 提取最后隐藏状态并获取最后一个token的嵌入
            last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

            # 获取批次中每个查询的最后一个非填充token的嵌入
            attention_mask = inputs["attention_mask"]
            sequence_lengths = attention_mask.shape[1] - 1  # 适用于 left padding
            batch_size_actual = last_hidden_state.shape[0]

            # 提取批次中每个序列的最后token嵌入
            last_token_embeddings = last_hidden_state[torch.arange(batch_size_actual, device="cuda"), sequence_lengths]

            # 存储嵌入 (移动到 CPU 并转换为 float32 以用于 safetensors)
            for j, query in enumerate(batch_queries):
                # Send CPU numpy arrays over the queue to avoid torch tensor
                # shared-memory file descriptor handling that can break once
                # the worker exits.
                embedding = last_token_embeddings[j].cpu().float().numpy()
                embeddings[query.strip()] = embedding

    print(f"GPU {device} 为 {len(embeddings)} 个查询生成了嵌入")
    queue.put(embeddings)


def save_embeddings(embeddings: dict[str, torch.Tensor], output_path: str):
    """
    将嵌入保存到 safetensors 文件.

    Args:
        embeddings: 将查询字符串映射到嵌入的字典
        output_path: 保存输出文件的路径
    """
    # 使用 SQL 查询的 hash 值作为键
    safe_tensors_dict = {}

    for query, embedding in embeddings.items():
        # 生成 SQL 查询的 MD5 hash 作为键
        query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()

        # 确保键是唯一的
        if query_hash in safe_tensors_dict:
            raise ValueError(f"Hash 冲突: {query_hash} 对应多个不同的查询 {query} 和 {safe_tensors_dict[query_hash]}")

        safe_tensors_dict[query_hash] = embedding

    # 保存嵌入
    print(f"正在保存嵌入到: {output_path}")
    save_file(safe_tensors_dict, output_path)
    print(f"已保存 {len(safe_tensors_dict)} 个嵌入")


def main():
    parser = argparse.ArgumentParser(description="使用语言模型为 SQL 查询生成嵌入")
    parser.add_argument("input_file", help="输入 SQL 文件的路径")
    parser.add_argument("--gpu", nargs="+", type=int, default=[0], help="要使用的 GPU 设备 ID 列表, 使用空格指定多个")
    parser.add_argument(
        "--output",
        "-o",
        help="safetensors 文件的输出路径 (默认: 与输入文件同名但扩展名为 .safetensors)",
    )
    parser.add_argument(
        "--model-path",
        "-m",
        default="models/qwen2.5-7b-instruct",
        help="模型目录路径 (默认: models/qwen2.5-7b-instruct)",
    )
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="每个 GPU 每次生成的批次大小 (默认: 64)")

    args = parser.parse_args()

    # 验证输入文件
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件 '{args.input_file}' 未找到")
        sys.exit(1)

    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input_file)
        output_path = str(input_path.with_suffix(".safetensors"))

    # 验证模型路径
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径 '{args.model_path}' 未找到")
        sys.exit(1)

    try:
        # 解析 SQL 查询
        print(f"正在从以下位置读取 SQL 查询: {args.input_file}")
        queries = read_file(args.input_file)

        if not queries:
            print("错误: 在输入文件中未找到有效的 SQL 查询")
            sys.exit(1)

        print(f"找到 {len(queries)} 个 SQL 查询")

        # 生成嵌入
        mp.set_start_method("spawn", force=True)
        processes = []
        q: "mp.SimpleQueue[dict[str, object]]" = mp.SimpleQueue()
        for i in range(len(args.gpu)):
            set_env_devices(os.environ, args.gpu[i])
            p = mp.Process(
                target=generate_embeddings,
                args=(q, args.gpu[i], queries[i :: len(args.gpu)], args.model_path, args.batch_size),
            )
            p.start()
            processes.append(p)

        # 收集所有进程的嵌入
        embeddings: dict[str, torch.Tensor] = {}
        for _ in range(len(args.gpu)):
            part = q.get()
            for query, embedding in part.items():
                embeddings[query] = torch.as_tensor(embedding)

        for p in processes:
            p.join()
        # 保存嵌入
        save_embeddings(embeddings, output_path)

        print("成功生成 SQL 嵌入!")

    except Exception as e:
        print(f"错误: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

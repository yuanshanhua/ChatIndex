import argparse
import multiprocessing as mp
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd
from safetensors.torch import save_file

from index_advisor.db import DBOption, init_worker_connections, worker_connection
from index_advisor.ia_logging import log_to_file, logger
from index_advisor.mm import show_embedding_info
from index_advisor.mm.tapas import TapasEmbeddingGenerator
from index_advisor.schemas import schemas
from index_advisor.workload import Workload, load_workloads, resolve_databases


warnings.filterwarnings(
    "ignore", category=FutureWarning, message="Series.__getitem__ treating keys as positions is deprecated"
)

# 全局只读缓存, 通过 fork 让子进程复用采样数据
_SAMPLE_STORE: dict[str, pd.DataFrame] = {}
_EMBEDDER: TapasEmbeddingGenerator | None = None


def _collect_required_columns(workloads: Iterable[Workload]) -> dict[str, dict[str, set[str]]]:
    """聚合所有 workload 涉及的表与列, 避免重复采样."""

    required: dict[str, dict[str, set[str]]] = {}
    for workload in workloads:
        for table_name, columns in workload.tables_columns.items():
            required.setdefault(workload.db, {}).setdefault(table_name, set()).update(columns)
    return required


def _sample_table_task(task: tuple[str, str, int]):
    """子进程任务: 从指定库/表采样 sample_count 行, 返回 DataFrame."""

    db_name, table_name, sample_count = task
    schema = schemas[db_name]
    table_schema = schema.tables[table_name]
    conn = worker_connection(db_name)
    samples = table_schema.get_sample_value(conn, sample_count)
    if not samples:
        return None
    df = pd.DataFrame(samples, columns=table_schema.ordered_column_names)
    return f"{db_name}.{table_name}", df


def _init_embed_worker(model_path: str, device: str | None, max_batch_rows: int):
    """在子进程中懒加载 TAPAS 模型, 避免父进程重复加载."""

    from index_advisor.mm.tapas import TapasEmbeddingGenerator

    global _EMBEDDER
    _EMBEDDER = TapasEmbeddingGenerator(model_path if model_path else "models/tapas-base", device, max_batch_rows)


def _embed_table_task(task: tuple[str, str, tuple[str, ...]]):
    """子进程任务: 使用已加载的模型生成表级嵌入."""

    embedding_key, table_key, columns = task
    df = _SAMPLE_STORE.get(table_key)
    if df is None:
        return None

    if _EMBEDDER is None:
        raise RuntimeError("TAPAS 模型未初始化")

    # 仅保留需要的列
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"表 {table_key} 缺少列: {missing}")
        table_df = df[list(columns)].copy()
    else:
        table_df = df

    embedding = _EMBEDDER.generate_table_embedding(table_df, question="")
    return embedding_key, embedding.cpu()


def main():
    parser = argparse.ArgumentParser(
        description="生成表格列嵌入", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--workload", "-w", type=str, default="", help="可选的 workload 文件路径")
    parser.add_argument("--algo", type=str, default="tapas", help="列嵌入算法")
    parser.add_argument("output", type=str, help="输出文件路径")
    parser.add_argument("-n", "--sample-count", type=int, default=1000, help="单个表的采样数量")
    parser.add_argument("--desc", type=str, default="", help="描述信息")
    parser.add_argument("--model_path", type=str, default="", help="列嵌入模型路径 (可选，默认使用内置路径)")
    parser.add_argument("--base_model_name", type=str, default="", help="TaBERT 基础模型名称 (可选，默认使用内置名称)")
    parser.add_argument("--db-workers", "--dbw", type=int, default=32, help="采样阶段的并发进程数")
    parser.add_argument("--model-workers", "--mw", type=int, default=32, help="生成表嵌入的并发进程数")
    parser.add_argument(
        "--max-batch-rows", "--maxb", type=int, default=200, help="TAPAS 单次处理的最大行数. 超过将分批生成后取均值"
    )
    parser.add_argument("--debug", action="store_true", help="输出更多日志信息")
    tabert = parser.add_argument_group("TaBERT", "TaBERT 模型相关选项")
    tabert.add_argument("--base", action="store_true", help="使用 base model")
    tabert.add_argument("--k3", action="store_true", default=True, help="使用 vertical attention (k3 model)")
    DBOption.add_arg_group_to(parser)

    args = parser.parse_args()
    if not args.output.endswith(".safetensors"):
        args.output += ".safetensors"
    log_to_file("gen_column_embeddings", "DEBUG" if args.debug else "INFO", args.desc)

    embeddings = {}

    workload_mode = bool(args.workload)
    workloads: list[Workload] = []
    workload_path: Path | None = None

    if workload_mode:
        if args.algo != "tapas":
            raise ValueError("当前仅支持使用 TAPAS 生成 workload 子集表嵌入")

        workload_path = Path(args.workload)
        if not workload_path.exists():
            raise FileNotFoundError(f"未找到 workload 文件: {workload_path}")

        workloads = load_workloads(workload_path)
        logger.info(f"loaded {len(workloads)} workloads from {workload_path}")

        args.db = list(resolve_databases(workloads))
        logger.info(f"自动添加 workload 相关数据库: {args.db}")
        db_option = DBOption.from_args(args)

        required_tables = _collect_required_columns(workloads)
        if not required_tables:
            raise RuntimeError("workload 中未找到任何表/列信息")

        # 1) 并行采样
        sample_tasks = [(db, table, args.sample_count) for db, tables in required_tables.items() for table in tables]
        sample_workers = max(1, args.db_workers)
        sample_ctx = mp.get_context("fork")
        opt_dict = db_option.to_dict() | {"databases": sorted(required_tables)}

        logger.info(f"并行采样 {len(sample_tasks)} 张表, 进程数 {sample_workers}")
        with sample_ctx.Pool(
            processes=sample_workers, initializer=init_worker_connections, initargs=(opt_dict,)
        ) as pool:
            results = pool.map(_sample_table_task, sample_tasks)

        sample_store: dict[str, pd.DataFrame] = {}
        for item in results:
            if not item:
                continue
            key, df = item
            sample_store[key] = df
        if not sample_store:
            raise RuntimeError("采样阶段未获取到任何表数据")
        logger.info(f"完成采样: {len(sample_store)} 张表")

        # 2) 构造去重的嵌入任务
        embed_tasks: list[tuple[str, str, tuple[str, ...]]] = []
        seen_keys: set[str] = set()
        for workload in workloads:
            workload_db = workload.db
            for table_name, columns in workload.tables_columns.items():
                if not columns:
                    logger.warning(f"workload {workload.id} 的表 {table_name} 没有列, 跳过")
                    continue
                sorted_columns = tuple(sorted(columns))
                embedding_key = "+".join(f"{workload_db}.{table_name}.{col}" for col in sorted_columns)
                if embedding_key in seen_keys:
                    continue
                seen_keys.add(embedding_key)
                table_key = f"{workload_db}.{table_name}"
                embed_tasks.append((embedding_key, table_key, sorted_columns))

        if not embed_tasks:
            raise RuntimeError("未生成任何嵌入任务")

        # 3) 并行生成嵌入
        embed_workers = max(1, args.model_workers)
        embed_ctx = mp.get_context("fork")
        global _SAMPLE_STORE
        _SAMPLE_STORE = sample_store

        logger.info(f"并行生成嵌入 {len(embed_tasks)} 个, 进程数 {embed_workers}")
        with embed_ctx.Pool(
            processes=embed_workers,
            initializer=_init_embed_worker,
            initargs=(args.model_path, None, args.max_batch_rows),
        ) as pool:
            for res in pool.imap_unordered(_embed_table_task, embed_tasks):
                if not res:
                    continue
                key, embedding = res
                embeddings[key] = embedding

        if not embeddings:
            raise RuntimeError("未生成任何 workload 表嵌入")
    else:
        if args.algo == "tapas":
            generator = TapasEmbeddingGenerator(args.model_path if args.model_path else "models/tapas-base")
        else:
            raise ValueError(f"不支持的列嵌入算法: {args.algo}")

        db_option = DBOption.from_args(args)
        embeddings = generator.generate_column_embeddings_from_db(db_option, sample_count=args.sample_count)

    metadata = {
        "sample_count": str(args.sample_count),
        "database": ",".join(args.db),
        "version": "250519",
        "desc": args.desc,
        "algo": args.algo,
    }
    if workload_mode:
        metadata.update(
            {
                "workload": args.workload,
                "workload_count": str(len(workloads)),
                "mode": "table",
            }
        )
    logger.info(f"Metadata: {metadata}")
    # 保存列嵌入
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    show_embedding_info(embeddings, logger.info)
    save_file(embeddings, output_path, metadata)
    logger.info(f"已生成并保存 {len(embeddings)} 个列嵌入到 {output_path}")


if __name__ == "__main__":
    main()

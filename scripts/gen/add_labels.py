from pathlib import Path

from index_advisor.dataset import load_index_eab_res
from index_advisor.db import DBOption
from index_advisor.ia_logging import log_to_file, logger
from index_advisor.workload import Workload, load_workloads, save_workloads


logger = logger.getChild("gen")


def run():
    import argparse

    parser = argparse.ArgumentParser(
        description="根据 index_eab 的索引推荐结果填充 workload 文件中的 labels 字段, 以用于 SFT. "
        "index_eab 对某些 workload 可能未生成有效索引, 这些 workload 会被跳过, 因此输出文件中 workload 数量小于等于输入文件.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("workload_file", help="workload 文件路径")
    parser.add_argument("input_json", help="index_eab 输出的 json 文件路径")
    parser.add_argument("--name", help="输出文件名. 全名为 {name}.labeled.json", default="")
    parser.add_argument("--desc", help="任务描述", type=str, default="生成 sft prompt")
    parser.add_argument("--max-len", help="限制生成的 prompt 最大长度", type=int, default=-1)
    parser.add_argument("--debug", help="调试模式", action="store_true")
    DBOption.add_arg_group_to(parser)
    args = parser.parse_args()
    db_option = DBOption.from_args(args)

    workload_file = Path(args.workload_file)
    name = workload_file.stem if not args.name else args.name
    output_file = workload_file.parent / f"{name}.labeled.json"

    log_to_file("add_labels", "DEBUG" if args.debug else "INFO", args.desc)

    workloads = load_workloads(workload_file)
    workloads_dict = {(w.db, frozenset(w.sqls)): w for w in workloads}

    labeled_workloads = load_index_eab_res(args.input_json, db_option.schema)

    merged_workloads: list[Workload] = []
    # 根据 db + sqls 匹配两个文件中的 workload
    for sqls, labels in labeled_workloads:
        key = (db_option.db, frozenset(sqls))
        if not labels:
            logger.warning(f"index_eab 未对 {key} 生成有效索引, 跳过")
            continue
        if key not in workloads_dict:
            logger.warning(f"工作负载 {key} 在 {workload_file} 中未找到, 跳过")
            continue
        w = workloads_dict[key]
        if w.labels is not None:
            logger.warning(f"工作负载 {key} 已包含 labels 字段, 将被覆盖")
        if w.label_overrides:
            logger.warning(f"工作负载 {key} 已包含 label_overrides 字段, 将被清空")
        w.labels = labels
        w.label_overrides.clear()
        merged_workloads.append(w)

    save_workloads(output_file, merged_workloads)


if __name__ == "__main__":
    run()

"""
从 pg 获取 db schema 并生成 Python模块
"""

import argparse
from pathlib import Path

from index_advisor.db import DBOption
from index_advisor.ia_logging import log_to_file, logger
from index_advisor.schemas import DatabaseSchema


def update_init_file(db_name: str):
    """更新__init__.py文件，添加导入语句"""
    init_path = Path("index_advisor/schemas/__init__.py")
    if not init_path.exists():
        logger.warning(f"__init__.py文件不存在: {init_path}，无法自动更新")
        return

    # 读取现有内容
    with open(init_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 构造导入语句
    import_line = f"from .{db_name} import {db_name}_schema"
    schemas_line = f'schemas["{db_name}"] = {db_name}_schema'

    # 检查是否已存在
    if import_line in content and schemas_line in content:
        logger.info(f"Schema {db_name} 已在__init__.py中定义，无需更新")
        return

    # 追加导入语句
    with open(init_path, "a", encoding="utf-8") as f:
        f.write(f"\n{import_line}\n{schemas_line}\n")

    logger.info(f"已更新__init__.py，添加了对{db_name}的导入")


def main():
    parser = argparse.ArgumentParser(
        description="从数据库获取 schema 信息, 并生成 Python Schema 定义. 支持一次处理多个数据库.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", action="store_true", help="打印详细信息")
    parser.add_argument("-o", "--output-dir", default="index_advisor/schemas", help="输出目录")
    parser.add_argument("--add-init", action="store_true", help="更新__init__.py文件, 添加导入语句")
    parser.add_argument("--debug", help="调试模式", action="store_true")
    DBOption.add_arg_group_to(parser)
    args = parser.parse_args()
    db_option = DBOption.from_args(args, raise_exception=False)
    # 设置日志
    log_to_file("schema", "DEBUG" if args.debug else "INFO", f"获取 schema: {db_option.databases}")
    for db in db_option.databases:
        # 检查是否覆盖
        module_path = Path(args.output_dir) / f"{db}.py"
        if module_path.exists():
            s = input(f"文件已存在: {module_path}, 是否覆盖? (y/n): ")
            if s.strip().lower() != "y":
                continue
        # 获取 schema
        schema = DatabaseSchema.from_connection(db, db_option.connections[db])
        schema.sample_large_columns(db_option.connections[db], sample_count=5, max_display_length=100)
        # 保存为 Python 模块
        with open(module_path, "w", encoding="utf-8") as f:
            f.write(schema.to_python())
        logger.info(f"Schema 已保存: {module_path}")
        # 更新 __init__.py
        if args.add_init:
            update_init_file(schema.name)


if __name__ == "__main__":
    main()

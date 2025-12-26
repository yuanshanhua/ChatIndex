#!/bin/bash

# 设置默认值
DIRECTORY="."
PATTERN="*.dat"
DRY_RUN=false
RECURSIVE=false

# 显示帮助信息
show_help() {
    cat << EOF
用法: $0 [选项] [目录]

移除指定文件中每一行最后的 | 符号

选项:
    -h, --help      显示此帮助信息
    -d, --dry-run   仅显示将要处理的文件，不实际修改
    -r, --recursive 递归处理子目录
    -p, --pattern   文件匹配模式 (默认: *.dat)

参数:
    目录            要处理的目录路径 (默认: 当前目录)

示例:
    $0                                    # 处理当前目录的 *.dat 文件
    $0 /path/to/data                      # 处理指定目录的 *.dat 文件
    $0 -r /path/to/data                   # 递归处理指定目录的 *.dat 文件
    $0 -d                                 # 演习模式，不实际修改文件
    $0 -p "*.txt" /path/to/data          # 处理指定目录的 *.txt 文件

EOF
}

# 处理命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -r|--recursive)
            RECURSIVE=true
            shift
            ;;
        -p|--pattern)
            PATTERN="$2"
            shift 2
            ;;
        -*)
            echo "错误: 未知选项 $1" >&2
            echo "使用 $0 --help 查看帮助信息" >&2
            exit 1
            ;;
        *)
            DIRECTORY="$1"
            shift
            ;;
    esac
done

# 检查目录是否存在
if [[ ! -d "$DIRECTORY" ]]; then
    echo "错误: 目录 '$DIRECTORY' 不存在" >&2
    exit 1
fi

# 查找文件
if [[ "$RECURSIVE" == true ]]; then
    # 递归查找
    mapfile -t FILES < <(find "$DIRECTORY" -name "$PATTERN" -type f)
else
    # 仅在指定目录查找
    mapfile -t FILES < <(find "$DIRECTORY" -maxdepth 1 -name "$PATTERN" -type f)
fi

# 检查是否找到文件
if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "在目录 '$DIRECTORY' 中没有找到匹配 '$PATTERN' 的文件"
    exit 0
fi

echo "找到 ${#FILES[@]} 个文件:"
for file in "${FILES[@]}"; do
    echo "  - $file"
done

# 如果是演习模式，直接退出
if [[ "$DRY_RUN" == true ]]; then
    echo
    echo "这是演习模式，没有文件被修改。"
    exit 0
fi

# 确认操作
echo
read -p "是否继续处理这 ${#FILES[@]} 个文件? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "操作已取消。"
    exit 0
fi

# 处理文件
total_files=${#FILES[@]}
total_lines_processed=0
total_lines_modified=0

echo
echo "开始处理 $total_files 个文件..."

for i in "${!FILES[@]}"; do
    file="${FILES[$i]}"
    echo
    echo "[$((i+1))/$total_files] 正在处理文件: $file"
    
    # 检查文件是否可写
    if [[ ! -w "$file" ]]; then
        echo "  警告: 文件不可写，跳过"
        continue
    fi
    
    # 统计总行数
    lines_in_file=$(wc -l < "$file")
    total_lines_processed=$((total_lines_processed + lines_in_file))
    
    # 统计以 | 结尾的行数
    lines_with_pipe=$(grep -c '|$' "$file" || true)
    total_lines_modified=$((total_lines_modified + lines_with_pipe))
    
    # 移除末尾的 |
    sed -i 's/|$//' "$file"
    
    echo "  - 总行数: $lines_in_file"
    echo "  - 修改行数: $lines_with_pipe"
done

echo
echo "处理完成!"
echo "总计:"
echo "  - 处理文件数: $total_files"
echo "  - 处理行数: $total_lines_processed"
echo "  - 修改行数: $total_lines_modified"

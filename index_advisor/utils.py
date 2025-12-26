import hashlib
from pathlib import Path

from transformers.trainer_utils import get_last_checkpoint

from .workload import Workload


def compute_workloads_summary(workloads: list[Workload]) -> str:
    """生成数据集摘要, 以验证拆分一致性."""
    hasher = hashlib.sha256()
    for w in workloads:
        for sql in w.sqls:
            hasher.update(sql.encode("utf-8"))
    return hasher.hexdigest()


def parse_ranges(s: str, keep_dup: bool = False) -> list[int]:
    """解析范围字符串, 允许逗号分隔的数字, 或者 start:end:step, 或者两者的组合. 保序且默认去重."""
    res = []
    for part in s.split(","):
        if ":" in part:
            parts = part.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"无效的范围格式: {part}")
            start, end, step = parts if len(parts) == 3 else (parts[0], parts[1], 1)
            start = int(start)
            end = int(end)
            step = int(step)
            if start > end:
                raise ValueError(f"start 必须小于 end: {part}")
            res.extend(list(range(start, end + 1, step)))
        else:
            res.append(int(part))
    return res if keep_dup else list(dict.fromkeys(res))


def resolve_ckpt(p: str | Path) -> str:
    """解析检查点路径, 如果 p 以 /last 结尾, 则返回最新检查点路径."""
    path = Path(p)
    if path.name == "last":
        path = path.parent
        last_ckpt = get_last_checkpoint(path)
        if last_ckpt is None:
            raise ValueError(f"目录中没有找到检查点: {p}")
        return last_ckpt
    else:
        return str(path)

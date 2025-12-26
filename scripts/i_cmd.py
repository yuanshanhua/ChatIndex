import argparse
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import yaml
from transformers import HfArgumentParser


BASE_DIR = "saves/multi_db"
SFT_PJ_CONFIG = "config/sft-projector.yaml"
SFT_CONFIG = "config/lib-sft.yaml"
RL_CONFIG = "config/train-qwen-mm.yaml"
EVAL_CONFIG = "config/eval-qwen.yaml"
DB_NAME = "dsb"
DB_PASSWORD = "your_password"
SQL_EMBEDDINGS = "workloads/index_eab/dsb_7000.safetensors"
COL_EMBEDDINGS = ""
ENABLE_SQL_EMBEDDINGS = True
ENABLE_COL_EMBEDDINGS = True
GROUP_SIZE = 3
EVAL_SET_PERCENT = 0.1
EVAL_BATCH_SIZE = 16
GEN_BATCH_SIZE = 16
FULL_EVAL = False

CONFIG_FILENAME = "i_cmd.yaml"


def parse_multi_values(value: str | List[str] | None) -> List[str]:
    """Split whitespace-separated strings into a list of tokens."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    stripped = value.strip()
    if not stripped:
        return []
    return shlex.split(stripped)


def build_db_args(config: "CommandConfig") -> List[str]:
    raw_value = config.db_name if config.db_name is not None else ""
    names = parse_multi_values(raw_value)
    if names:
        return ["--db", *names]
    return ["--db", raw_value]


@dataclass
class CommandConfig:
    sft_pj_config: str = SFT_PJ_CONFIG
    sft_config: str = SFT_CONFIG
    rl_config: str = RL_CONFIG
    eval_config: str = EVAL_CONFIG
    db_name: str = DB_NAME
    db_password: str = DB_PASSWORD
    sql_embeddings: str = SQL_EMBEDDINGS
    col_embeddings: str = COL_EMBEDDINGS
    enable_sql_embeddings: bool = ENABLE_SQL_EMBEDDINGS
    enable_col_embeddings: bool = ENABLE_COL_EMBEDDINGS
    group_size: int = GROUP_SIZE
    eval_set_percent: float = EVAL_SET_PERCENT
    eval_batch_size: int = EVAL_BATCH_SIZE
    gen_batch_size: int = GEN_BATCH_SIZE
    full_eval: bool = FULL_EVAL


def load_or_create_config(base_dir: Path) -> CommandConfig:
    base_dir.mkdir(parents=True, exist_ok=True)
    config_path = base_dir / CONFIG_FILENAME
    parser = HfArgumentParser(CommandConfig)  # type: ignore[arg-type]
    raw_config: Dict[str, Any] = {}

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"配置文件 {config_path} 应该是一个字典结构")
        raw_config = {str(k): v for k, v in loaded.items()}

    (config,) = parser.parse_dict(raw_config)
    serialized = asdict(config)

    if (not config_path.exists()) or raw_config != serialized:
        with config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(serialized, fh, allow_unicode=False, sort_keys=False)
        print(f"# 提示: 已写入配置文件 {config_path}", file=sys.stderr)

    return config


@dataclass
class Context:
    iteration: int
    step: str
    base_dir: Path
    gpu: str
    config: CommandConfig

    @property
    def dataset_path(self) -> str:
        return str(self.base_dir / f"workload.{self.iteration}.json")

    @property
    def prev_dataset_path(self) -> str:
        prev_iter = self.iteration - 1 if self.iteration > 0 else 0
        return str(self.base_dir / f"workload.{prev_iter}.json")


def prev_rl_adapter(ctx: Context) -> str:
    if ctx.iteration > 0:
        return str(ctx.base_dir / f"rl.{ctx.iteration - 1}" / "checkpoint-⚠️")
    return "⚠️ projector_or_prev_rl_dir"


def sft_adapter(ctx: Context) -> str:
    if ctx.iteration > 0:
        return str(ctx.base_dir / f"sft.{ctx.iteration}" / "checkpoint-⚠️")
    return "⚠️ sft_ckpt_dir"


def format_command(tokens: List[str], env: str | None = None) -> str:
    quoted = [shlex.quote(t) for t in tokens]
    lines: List[str] = []
    current: List[str] = []
    for token in quoted:
        candidate = " ".join(current + [token])
        if current and len(candidate) > 100:
            lines.append(" ".join(current) + " \\")
            current = [token]
        else:
            current.append(token)
    if current:
        lines.append(" ".join(current))
    if env:
        if lines:
            lines[0] = f"{env} {lines[0]}"
        else:
            lines.append(env)
    return "\n".join(lines)


def projector_projector_ckpt(adapter_dir: str) -> str:
    if adapter_dir.startswith("⚠️"):
        return "⚠️ projector_safetensors"
    return f"{adapter_dir}/projector.safetensors"


def get_mm_args(config: CommandConfig, trainable: bool = False, projector_ckpt: str | None = None) -> List[str]:
    args = []
    sql_embeddings = parse_multi_values(config.sql_embeddings)
    col_embeddings = parse_multi_values(config.col_embeddings)
    if config.enable_sql_embeddings:
        args.append("--smm")
        if sql_embeddings:
            args.extend(["--smm-efs", *sql_embeddings])
        if trainable:
            args.append("--smm-pt")
        if projector_ckpt:
            args.extend(["--smm-pckpt", projector_ckpt])

    if config.enable_col_embeddings:
        args.append("--cmm")
        if col_embeddings:
            args.extend(["--cmm-efs", *col_embeddings])
        if trainable:
            args.append("--cmm-pt")
        if projector_ckpt:
            args.extend(["--cmm-pckpt", projector_ckpt])
    return args


def build_eval_dataset_args(ctx: Context, workload: str) -> List[str]:
    """Return eval dataset CLI args based on whether full_eval is enabled."""
    workload_path = str(workload)
    if ctx.config.full_eval:
        return ["--eval-data-file", workload_path]
    return ["--eval-set-percent", str(ctx.config.eval_set_percent)]


def build_projector(ctx: Context) -> Tuple[str, List[str], str]:
    assert ctx.iteration == 1, "仅第 1 轮需要 train projector only"
    workload = ctx.prev_dataset_path
    output_dir = ctx.base_dir / f"sft.pj.{ctx.iteration}"
    mm_args = get_mm_args(ctx.config, trainable=True)
    eval_args = build_eval_dataset_args(ctx, workload)
    db_args = build_db_args(ctx.config)
    tokens = (
        [
            "uv",
            "run",
            "scripts/multi_sft.py",
            workload,
            ctx.config.sft_pj_config,
        ]
        + eval_args
        + [
            "--output-dir",
            str(output_dir),
            "--mm-flm",
        ]
        + mm_args
        + [
            "--epochs",
            "3",
            "--batch-size",
            "4",
            "--learning-rate",
            "1e-3",
            "--group-size",
            str(ctx.config.group_size),
            "--save-steps",
            "100",
            "--log-steps",
            "10",
        ]
    )
    tokens += db_args
    tokens += ["--dbpswd", ctx.config.db_password]
    label = "步骤1 - sft projector"
    command = format_command(tokens, env=f"CUDA_VISIBLE_DEVICES={ctx.gpu}")
    return command, [], label


def build_sft(ctx: Context) -> Tuple[str, List[str], str]:
    prev_iter = ctx.iteration - 1 if ctx.iteration > 0 else 0
    workload = ctx.prev_dataset_path
    adapter_dir = prev_rl_adapter(ctx)
    projector_ckpt = projector_projector_ckpt(adapter_dir)
    output_dir = ctx.base_dir / f"sft.{ctx.iteration}"
    mm_args = get_mm_args(ctx.config, trainable=True, projector_ckpt=projector_ckpt)
    eval_args = build_eval_dataset_args(ctx, workload)
    db_args = build_db_args(ctx.config)
    tokens = (
        [
            "uv",
            "run",
            "scripts/multi_sft.py",
            workload,
            ctx.config.sft_config,
            "--adapter-dir",
            adapter_dir,
        ]
        + eval_args
        + [
            "--output-dir",
            str(output_dir),
        ]
        + mm_args
        + [
            "--epochs",
            "5",
            "--batch-size",
            "4",
            "--learning-rate",
            "1e-4",
            "--group-size",
            str(ctx.config.group_size),
            "--save-steps",
            "100",
            "--log-steps",
            "10",
        ]
    )
    tokens += db_args
    tokens += ["--dbpswd", ctx.config.db_password]
    notes = [f"使用第 {prev_iter} 轮数据集"]
    label = "步骤2 - 全量 sft"
    command = format_command(tokens, env=f"CUDA_VISIBLE_DEVICES={ctx.gpu}")
    return command, notes, label


def build_rl(ctx: Context) -> Tuple[str, List[str], str]:
    prev_iter = ctx.iteration - 1 if ctx.iteration > 0 else 0
    workload = ctx.prev_dataset_path
    adapter_dir = sft_adapter(ctx)
    projector_ckpt = projector_projector_ckpt(adapter_dir)
    output_dir = ctx.base_dir / f"rl.{ctx.iteration}"
    mm_args = get_mm_args(ctx.config, trainable=True, projector_ckpt=projector_ckpt)
    eval_args = build_eval_dataset_args(ctx, workload)
    db_args = build_db_args(ctx.config)
    tokens = (
        [
            "uv",
            "run",
            "scripts/train.py",
            workload,
            ctx.config.rl_config,
            "--adapter-dir",
            adapter_dir,
        ]
        + eval_args
        + [
            "--eval-batch-size",
            str(ctx.config.eval_batch_size),
            "--gen-batch-size",
            str(ctx.config.gen_batch_size),
        ]
        + mm_args
        + db_args
        + [
            "--dbpswd",
            ctx.config.db_password,
            "--output-dir",
            str(output_dir),
            "--group-size",
            str(ctx.config.group_size),
        ]
    )
    notes = [f"使用第 {prev_iter} 轮数据集"]
    label = "步骤3 - RL"
    command = format_command(tokens, env=f"CUDA_VISIBLE_DEVICES={ctx.gpu}")
    return command, notes, label


def build_eval(ctx: Context) -> Tuple[str, List[str], str]:
    eval_output = ctx.base_dir / f"eval.{ctx.iteration}.json"
    adapter_dir = ctx.base_dir / f"rl.{ctx.iteration}" / "checkpoint-0"
    mm_args = get_mm_args(ctx.config, trainable=False)
    db_args = build_db_args(ctx.config)
    tokens = (
        [
            "uv",
            "run",
            "scripts/eval.py",
            ctx.prev_dataset_path,
            str(eval_output),
            "-b",
            "32",
        ]
        + db_args
        + [
            "--dbpswd",
            ctx.config.db_password,
        ]
        + mm_args
        + [
            "--adapter-dir",
            str(adapter_dir),
            "--ckpt",
            "all",
            "--gpu",
            ctx.gpu,
            ctx.config.eval_config,
            "--round",
            "10",
            "--group-size",
            str(ctx.config.group_size),
        ]
    )
    notes = [
        "根据实际评估需求修改 --ckpt 和 --gpu",
        "确保 eval 输出文件名不会覆盖历史结果",
    ]
    label = "步骤4 - eval"
    command = format_command(tokens)
    return command, notes, label


def build_update(ctx: Context) -> Tuple[str, List[str], str]:
    prev_iter = ctx.iteration - 1 if ctx.iteration > 0 else 0
    source_dataset = ctx.prev_dataset_path
    target_dataset = ctx.dataset_path
    adapter_dir = ctx.base_dir / f"rl.{ctx.iteration}" / "checkpoint-0"
    mm_args = get_mm_args(ctx.config, trainable=False)
    db_args = build_db_args(ctx.config)
    tokens = (
        [
            "uv",
            "run",
            "scripts/update_labels.py",
            source_dataset,
            ctx.config.eval_config,
            "--output",
            target_dataset,
            "-b",
            "32",
            "--adapter-dir",
            str(adapter_dir),
            "--ckpt",
            "all",
            "--gpu",
            ctx.gpu,
            "--group-size",
            str(ctx.config.group_size),
        ]
        + mm_args
        + [
            *db_args,
            "--dbpswd",
            ctx.config.db_password,
            "--eval-workers",
            "8",
        ]
    )
    notes = [
        f"读取第 {prev_iter} 轮数据集 {source_dataset}，请确认与评估结果匹配。",
        f"新数据集写入 {target_dataset}，如有自定义目录请自行修改。",
        f"--adapter-dir 默认指向 {adapter_dir}，请选择实际的 checkpoint。",
        "如需筛选特定 checkpoint，请补充 --ckpt 选项 (例如 0 或 500:1000:100)。",
    ]
    label = "步骤5 - 更新数据集"
    command = format_command(tokens)
    return command, notes, label


STEP_BUILDERS: dict[str, Callable[[Context], Tuple[str, List[str], str]]] = {
    "projector": build_projector,
    "sft": build_sft,
    "rl": build_rl,
    "eval": build_eval,
    "update": build_update,
}


def normalize_step(step: str, iteration: int) -> str:
    cleaned = step.strip().lower()
    if "-" in cleaned:
        prefix, suffix = cleaned.split("-", 1)
        if prefix.isdigit() and int(prefix) != iteration:
            raise ValueError("传入的步数前缀与当前轮次不符")
        cleaned = suffix
    aliases = {
        "1": "projector",
        "projector": "projector",
        "2": "sft",
        "sft": "sft",
        "3": "rl",
        "rl": "rl",
        "ppo": "rl",
        "4": "eval",
        "evaluate": "eval",
        "eval": "eval",
        "5": "update",
        "update": "update",
        "labels": "update",
    }
    if cleaned not in aliases:
        raise ValueError(f"不支持的步骤标识: {step}")
    return aliases[cleaned]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成迭代训练算法指定轮次-步骤的训练命令模板.")
    parser.add_argument("--iteration", "-n", type=int, required=True, help="当前轮次 N")
    parser.add_argument("--step", "-s", required=True, help="步骤标识. 1-projector | 2-sft | 3-rl | 4-eval | 5-update")
    parser.add_argument("--base-dir", "-b", default=BASE_DIR, help="保存 checkpoint 与评估结果的基础目录")
    parser.add_argument("--gpu", default="0", help="使用的 GPU 编号")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    base_dir = Path(args.base_dir)
    config = load_or_create_config(base_dir)
    step = normalize_step(args.step, args.iteration)
    ctx = Context(
        iteration=args.iteration,
        step=step,
        base_dir=base_dir,
        gpu=args.gpu,
        config=config,
    )
    builder = STEP_BUILDERS[ctx.step]
    command, notes, label = builder(ctx)
    print(f"# 第 {ctx.iteration} 轮 {label} 命令模板")
    for note in notes:
        print(f"# 提示: {note}")
    print(command)
    print("# 提醒: 执行前请确认所有占位符已替换为实际值。")


if __name__ == "__main__":
    main()

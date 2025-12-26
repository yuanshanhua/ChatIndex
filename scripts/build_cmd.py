import os
import readline  # fallback tab completion
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from transformers.utils.import_utils import is_torch_npu_available


# Optional rich
try:
    from rich import print  # type: ignore
    from rich.panel import Panel  # type: ignore
    from rich.prompt import Confirm  # type: ignore
except Exception:  # pragma: no cover

    def print(*a, **k):  # type: ignore
        __builtins__["print"](*a, **k)

    Confirm = None  # type: ignore
    Panel = None  # type: ignore

# Optional prompt_toolkit
try:
    from prompt_toolkit import prompt  # type: ignore
    from prompt_toolkit.completion import Completer, Completion, PathCompleter  # type: ignore
except Exception:  # pragma: no cover
    prompt = None  # type: ignore
    PathCompleter = None  # type: ignore
    Completer = object  # type: ignore
    Completion = object  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKLOAD_DIR = ROOT / "workloads"
DEFAULT_COL_EMBED_DIR = ROOT / "embeddings"
DEFAULT_SQL_EMBED_DIR = ROOT / "workloads"
DEFAULT_CONFIG_DIR = ROOT / "config"
DEFAULT_SAVES_DIR = ROOT / "saves"

SFT_SCRIPT = "scripts/sft_projector.py"
RL_SCRIPT = "scripts/train.py"
MULTI_RL_SCRIPT = "scripts/multi_train.py"  # wrapper for multi-GPU RL


@dataclass
class BuildContext:
    task: str  # "sft" or "rl"
    workload_file: str = ""
    config_file: str = "config/train-qwen-mm.yaml"  # 可同时用于 sft / rl，留空则不传
    output_dir: str = ""
    adapter_dir: str = ""
    eval_set_percent: float = 0.1
    eval_batch_size: int = 16  # RL only
    gen_batch_size: Optional[int] = None  # RL only
    batch_size: int = 4  # SFT
    epochs: int = 20
    learning_rate: str = "1e-4"
    save_steps: Optional[int] = None  # SFT
    log_steps: Optional[int] = None  # SFT
    mm: bool = True
    mm2: bool = False
    mm_embedding_files: List[str] = field(default_factory=list)
    mm2_embedding_files: List[str] = field(default_factory=list)
    mm_projector_trainable: bool = True
    mm2_projector_trainable: bool = False
    # projector checkpoint (统一命名, 原 --mm-pckpt 已弃用)
    mm_projector_checkpoint: str = ""  # --mm-projector-checkpoint
    mm2_projector_checkpoint: str = ""  # --mm2-projector-checkpoint (统一, 原 --mm2-pckpt 已弃用)
    desc: str = ""
    dbs: List[str] = field(default_factory=list)  # RL: --db imdb tpch1g
    dbpswd: str = ""
    dbuser: str = ""  # used maybe future
    dbport: str = ""  # used maybe future
    reward: str = "cost"  # RL only
    debug: bool = False
    gpus: List[int] = field(default_factory=lambda: [0])  # RL multi-gpu list


# ---------------------- Helpers ----------------------


def _list_files(directory: Path, suffixes: Tuple[str, ...]) -> List[str]:
    if not directory.exists():
        return []
    res = []
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix in suffixes:
            try:
                rel = p.relative_to(ROOT).as_posix()
            except Exception:
                rel = str(p)
            res.append(rel)
    return sorted(res)


def discover_workloads() -> List[str]:
    return _list_files(DEFAULT_WORKLOAD_DIR, (".json", ".workload", ".workload.json"))


def discover_mm_embeddings() -> List[str]:
    return _list_files(DEFAULT_COL_EMBED_DIR, (".safetensors",))


def discover_mm2_embeddings() -> List[str]:
    return _list_files(DEFAULT_SQL_EMBED_DIR, (".safetensors",))


def discover_configs() -> List[str]:
    return _list_files(DEFAULT_CONFIG_DIR, (".yaml", ".yml"))


def discover_adapters() -> List[str]:
    # Heuristic: look for checkpoint-*/projector.safetensors or adapter_model.safetensors
    if not DEFAULT_SAVES_DIR.exists():
        return []
    res: List[str] = []
    for p in DEFAULT_SAVES_DIR.rglob("projector.safetensors"):
        try:
            res.append(p.parent.relative_to(ROOT).as_posix())
        except Exception:
            res.append(str(p.parent))
    for p in DEFAULT_SAVES_DIR.rglob("adapter_model.safetensors"):
        try:
            res.append(p.parent.relative_to(ROOT).as_posix())
        except Exception:
            res.append(str(p.parent))
    return sorted(set(res))


# ---------------------- Simple selection utilities ----------------------


def choose_from_list(title: str, items: List[str], multi: bool = False, default: Optional[str] = None) -> List[str]:
    if not items:
        return []
    if prompt is None:  # fallback simple text
        print(f"[bold]{title} (fallback 输入)[/bold]") if "bold" in dir() else print(title)
        for i, it in enumerate(items):
            print(f"  [{i}] {it}")
        sel = input(f"选择索引{'(多个用逗号)' if multi else ''}{' 默认:' + default if default else ''}: ").strip()
        if not sel and default:
            return [default]
        if multi:
            idxs = [int(s) for s in sel.split(",") if s.strip().isdigit()]
            return [items[i] for i in idxs if 0 <= i < len(items)]
        else:
            if not sel:
                return [items[0]]
            i = int(sel)
            return [items[i]]
    else:
        # minimal interactive with indices (still simple)
        from prompt_toolkit.shortcuts import checkboxlist_dialog, radiolist_dialog  # type: ignore

        if multi:
            result = checkboxlist_dialog(
                title=title,
                text="使用空格选择，回车确认",
                values=[(str(i), it) for i, it in enumerate(items)],
            ).run()
            if not result:
                return []
            return [items[int(r)] for r in result]
        else:
            result = radiolist_dialog(
                title=title,
                text="请选择一项",
                values=[(str(i), it) for i, it in enumerate(items)],
                default=str(items.index(default)) if default and default in items else None,
            ).run()
            if result is None:
                return []
            return [items[int(result)]]


# ---------------------- Input with completion ----------------------
if prompt is None:

    class _BasicPathCompleter:  # fallback dummy
        pass

    # setup readline path completion (Linux bash style)
    def _rl_path_completer(text, state):  # pragma: no cover - interactive
        """Return the next possible completion for 'text'."""
        line = readline.get_line_buffer()
        # Expand user ~ and env vars
        if not line:
            line = text
        cur = os.path.expanduser(text or "")
        dirname = cur if os.path.isdir(cur) else os.path.dirname(cur) or "."
        prefix = os.path.basename(cur)
        try:
            entries = os.listdir(dirname or ".")
        except Exception:
            entries = []
        matches = []
        for e in entries:
            if e.startswith(prefix):
                full = os.path.join(dirname, e)
                if os.path.isdir(full):
                    matches.append(os.path.join(dirname, e) + "/")
                else:
                    matches.append(os.path.join(dirname, e))
        matches.sort()
        if state < len(matches):
            return matches[state]
        return None

    try:  # register completer
        readline.set_completer_delims(" \t\n\"'`@$><=;|&{(")
        readline.set_completer(_rl_path_completer)
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass
else:

    class _BasicPathCompleter(Completer):  # type: ignore
        def get_completions(self, document, complete_event):  # pragma: no cover - UI path
            # If Completion is a stub (object), skip
            if Completion is object:  # type: ignore
                return
            text = document.text_before_cursor
            expanded = os.path.expanduser(text or ".")
            dirname = os.path.dirname(expanded) or "."
            try:
                for name in os.listdir(dirname):
                    full = os.path.join(dirname, name)
                    disp = name + "/" if os.path.isdir(full) else name
                    # prompt_toolkit.completion.Completion(text, start_position=...)
                    yield Completion(disp, start_position=0)  # type: ignore[arg-type]
            except Exception:
                return


def _to_rel(p: str) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except Exception:
        return p


def select_safetensor_file(base: str, title: str) -> str:
    """If base is a directory list *.safetensors and ask user to choose.
    Return relative selected path or original base if no files.
    """
    if not base:
        return base
    if os.path.isdir(base):
        files = []
        for p in Path(base).glob("*.safetensors"):
            try:
                files.append(p.relative_to(ROOT).as_posix())
            except Exception:
                files.append(p.as_posix())
        if files:
            sel = choose_from_list(title, sorted(files), multi=False)
            if sel:
                return sel[0]
            return files[0]
        else:
            print("[yellow]目录下未找到 *.safetensors, 使用原路径[/yellow]")
            return base
    return base


def ask_path(msg: str, default: str = "", must_exist: bool = True) -> str:
    base_msg = f"{msg}{' [' + default + ']' if default else ''}: "
    if prompt is None:
        val = input(base_msg).strip()
    else:  # use prompt_toolkit path completion
        completer = PathCompleter() if PathCompleter else _BasicPathCompleter()
        val = prompt(base_msg, completer=completer).strip()
    if not val and default:
        val = default
    val = os.path.expanduser(val)
    if must_exist and val and not os.path.exists(val):
        print(f"[yellow]警告: 路径 {val} 不存在[/yellow]")
    return _to_rel(val) if val else val


def ask(msg: str, default: Optional[str] = None) -> str:
    base_msg = f"{msg}{' [' + default + ']' if default is not None else ''}: "
    if prompt is None:
        val = input(base_msg).strip()
    else:
        val = prompt(base_msg).strip()
    if not val and default is not None:
        return default
    return val


def ask_bool(msg: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        v = ask(f"{msg} ({d})", "y" if default else "n").lower()
        if v in ("y", "yes"):
            return True
        if v in ("n", "no"):
            return False
        print("请输入 y 或 n")


# ---------------------- Build command ----------------------


def build_sft(ctx: BuildContext) -> List[str]:
    args: List[str] = ["uv", "run", SFT_SCRIPT, ctx.workload_file, ctx.config_file]
    if ctx.eval_set_percent:
        args += ["--eval-set-percent", str(ctx.eval_set_percent)]
    if ctx.output_dir:
        args += ["--output-dir", ctx.output_dir]
    if ctx.mm:
        args.append("--mm")
    if ctx.mm_embedding_files:
        args += ["--mm-embedding-files", *ctx.mm_embedding_files]
    if ctx.mm_projector_trainable:
        args.append("--mm-projector-trainable")
    if ctx.mm_projector_checkpoint:
        args += ["--mm-projector-checkpoint", ctx.mm_projector_checkpoint]
    if ctx.mm2:
        args.append("--mm2")
    if ctx.mm2_embedding_files:
        args += ["--mm2-embedding-files", *ctx.mm2_embedding_files]
    if ctx.mm2_projector_trainable:
        args.append("--mm2-projector-trainable")
    if ctx.mm2_projector_checkpoint:
        args += ["--mm2-projector-checkpoint", ctx.mm2_projector_checkpoint]
    if ctx.adapter_dir:
        args += ["--adapter-dir", ctx.adapter_dir]
    args += [
        "--epochs",
        str(ctx.epochs),
        "--batch-size",
        str(ctx.batch_size),
        "--learning-rate",
        str(ctx.learning_rate),
    ]
    if ctx.save_steps:
        args += ["--save-steps", str(ctx.save_steps)]
    if ctx.log_steps:
        args += ["--log-steps", str(ctx.log_steps)]
    if ctx.debug:
        args.append("--debug")
    return args


def build_rl(ctx: BuildContext) -> List[str]:
    nproc = len(ctx.gpus)
    env = []
    if ctx.gpus:
        if is_torch_npu_available():
            env.append(f"ASCEND_RT_VISIBLE_DEVICES={','.join(str(g) for g in ctx.gpus)}")
        else:
            env.append(f"CUDA_VISIBLE_DEVICES={','.join(str(g) for g in ctx.gpus)}")
    # 多卡使用 multi_train.py, 单卡保持 torchrun 兼容（或未来可改为直接 uv run train.py）
    if nproc > 1:
        base: List[str] = ["uv", "run", MULTI_RL_SCRIPT, ctx.workload_file, ctx.config_file]
    else:
        base = ["uv", "run", RL_SCRIPT, ctx.workload_file, ctx.config_file]
    args: List[str] = env + base
    if ctx.adapter_dir:
        args += ["--adapter-dir", ctx.adapter_dir]
    if ctx.output_dir:
        args += ["--output-dir", ctx.output_dir]
    if ctx.eval_set_percent:
        args += ["--eval-set-percent", str(ctx.eval_set_percent)]
    if ctx.eval_batch_size:
        args += ["--eval-batch-size", str(ctx.eval_batch_size)]
    if ctx.gen_batch_size:
        args += ["--gen-batch-size", str(ctx.gen_batch_size)]
    if ctx.mm:
        args.append("--mm")
    if ctx.mm_embedding_files:
        args += ["--mm-embedding-files", *ctx.mm_embedding_files]
    if ctx.mm_projector_trainable:
        args.append("--mm-projector-trainable")
    if ctx.mm_projector_checkpoint:
        # 对于 RL 也复用同名参数，含义相同
        args += ["--mm-projector-checkpoint", ctx.mm_projector_checkpoint]
    if ctx.mm2:
        args.append("--mm2")
    if ctx.mm2_embedding_files:
        args += ["--mm2-embedding-files", *ctx.mm2_embedding_files]
    if ctx.mm2_projector_trainable:
        args.append("--mm2-projector-trainable")
    if ctx.mm2_projector_checkpoint:
        args += ["--mm2-projector-checkpoint", ctx.mm2_projector_checkpoint]
    if ctx.dbs:
        args += ["--db", *ctx.dbs]
    if ctx.dbpswd:
        args += ["--dbpswd", ctx.dbpswd]
    if ctx.reward:
        args += ["--reward", ctx.reward]
    if ctx.debug:
        args.append("--debug")
    return args


# ---------------------- Interactive flow ----------------------


def interactive() -> None:
    print(
        Panel("Index Advisor 训练命令生成器", title="Train Command Builder")
        if Panel
        else "== Train Command Builder =="
    )
    task = ask("选择任务类型 (sft/rl)", "sft").lower()
    if task not in {"sft", "rl"}:
        print("任务类型无效, 退出")
        return
    ctx = BuildContext(task=task)

    # workload 选择
    ctx.workload_file = ask_path("输入 workload 文件路径", must_exist=True)

    cfgs = discover_configs()
    if cfgs:
        print(f"发现 {len(cfgs)} 个配置文件，可选择或留空使用默认/不传 (--config 可选)")
        for i, c in enumerate(cfgs[:12]):
            print(f"  [{i}] {c}")
        cfg_in = ask("输入配置序号或路径(留空跳过)", "")
        if cfg_in.isdigit():
            idx = int(cfg_in)
            if 0 <= idx < len(cfgs):
                ctx.config_file = cfgs[idx]
        elif cfg_in:
            ctx.config_file = _to_rel(os.path.expanduser(cfg_in))
    else:
        manual_cfg = ask_path("输入配置文件(留空跳过)", "", must_exist=False)
        if manual_cfg:
            ctx.config_file = manual_cfg

    # output dir
    default_out = f"saves/{'sft_projector' if task == 'sft' else 'ppo'}"
    ctx.output_dir = ask_path("输出目录", default_out, must_exist=False)

    # adapter: 直接输入，可留空跳过；展示发现的候选
    manual = ask_path("输入 adapter 目录(留空跳过)", "", must_exist=False)
    ctx.adapter_dir = manual or ""

    # 多模态选项
    ctx.mm = ask_bool("启用 mm?", True)
    if ctx.mm:
        emb = discover_mm_embeddings()
        if emb:
            sel = choose_from_list("选择 mm embedding (可多选)", emb, multi=True)
            ctx.mm_embedding_files = sel[:2]  # 常见两个
        ctx.mm_projector_trainable = ask_bool("mm projector 可训练?", True)
        # 直接输入 checkpoint 路径，留空表示不设置
        ck = ask_path("mm projector checkpoint 路径(文件或目录, 留空跳过)", "", must_exist=True)
        if ck:
            ctx.mm_projector_checkpoint = select_safetensor_file(ck, "选择 checkpoint 文件")

    ctx.mm2 = ask_bool("启用第二模态 mm2?", False)
    if ctx.mm2:
        emb2 = discover_mm2_embeddings()
        if emb2:
            sel2 = choose_from_list("选择 mm2 embedding (可多选)", emb2, multi=True)
            ctx.mm2_embedding_files = sel2[:2]
        ctx.mm2_projector_trainable = ask_bool("mm2 projector 可训练?", True)
        ck2 = ask_path("mm2 projector checkpoint 路径(文件或目录, 留空跳过)", "", must_exist=True)
        if ck2:
            ctx.mm2_projector_checkpoint = select_safetensor_file(ck2, "选择 mm2 checkpoint 文件")

    if task == "sft":
        ctx.epochs = int(ask("训练轮数", str(ctx.epochs)))
        ctx.batch_size = int(ask("batch size", str(ctx.batch_size)))
        ctx.learning_rate = ask("learning rate", ctx.learning_rate)
        ctx.eval_set_percent = float(ask("eval 集比例(0-1)", str(ctx.eval_set_percent)))
        ctx.save_steps = int(ask("save steps", "100"))
        ctx.log_steps = int(ask("log steps", "10"))
    else:  # RL
        gpu_str = ask("GPU 列表(逗号)", "0,1")
        ctx.gpus = [int(x) for x in gpu_str.split(",") if x.strip().isdigit()]
        ctx.eval_set_percent = float(ask("eval 集比例(0-1)", str(ctx.eval_set_percent)))
        ctx.eval_batch_size = int(ask("eval batch size", str(ctx.eval_batch_size)))
        gb = ask("gen batch size(PPO rollout)", "")
        if gb:
            ctx.gen_batch_size = int(gb)
        ctx.reward = ask("reward(cost/format/div)", ctx.reward)
        db_in = ask("数据库列表(空格分隔)", "imdb tpch1g")
        ctx.dbs = [d for d in db_in.split() if d]
        ctx.dbpswd = ask("数据库密码", "your_password")

    ctx.debug = ask_bool("启用 --debug?", False)
    ctx.desc = ask("描述(可选)", ctx.desc)

    # Build command
    if task == "sft":
        cmd_list = build_sft(ctx)
    else:
        cmd_list = build_rl(ctx)

    # Pretty output
    # Escape args with spaces
    def esc(a: str) -> str:
        return shlex.quote(a) if " " in a or "(" in a or ")" in a else a

    lines = []
    current = []
    for a in cmd_list:
        if len(" ".join(current + [a])) > 110:
            lines.append(" ".join(current) + " \\")
            current = [esc(a)]
        else:
            current.append(esc(a))
    if current:
        lines.append(" ".join(current))

    final_script = "\n".join(lines)

    header = f"# Generated command ({task})"
    if ctx.desc:
        header += f"\n# {ctx.desc}"
    print("\n" + header)
    print(final_script)

    if ask_bool(f"写入到 {ctx.output_dir}/cmd.sh ?", True):
        out_path = Path(ctx.output_dir) / "cmd.sh"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("#!/usr/bin/env bash\nset -euo pipefail\n" + header + "\n" + final_script + "\n")
        os.chmod(out_path, 0o755)
        print(f"已写入 {out_path}")


def main():  # pragma: no cover
    try:
        interactive()
    except KeyboardInterrupt:
        print("\n用户中断, 退出。")


if __name__ == "__main__":
    main()

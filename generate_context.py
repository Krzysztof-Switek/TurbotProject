"""Standalone context generator for TurbotProject.

Scans the whole project (pure static analysis, no code execution) and writes a
small set of Markdown files into ``context_files/`` that summarise where things
are and why, so a new working session can be bootstrapped without reading every
source file.

Stdlib-only by design (uses :mod:`ast`) so it runs even when the project
virtualenv is broken.

Usage::

    python generate_context.py [--root <dir>] [--out <dir>]

Outputs (written to ``<root>/context_files``):
    00_INDEX.md         entry point: stats, entry points, quick navigation
    01_FILE_MAP.md      directory tree + per-file purpose table
    02_CODE_MAP.md      AST detail: classes / methods / functions + docstrings
    03_DEPENDENCIES.md  internal import graph + external libraries
    04_ENVIRONMENT.md   requirements, configs, planning docs inventory
"""

from __future__ import annotations

import argparse
import ast
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

EXCLUDE_DIRS = {
    ".git", ".venv", ".venv-web", ".idea", "__pycache__", "runs",
    "output_crops", "test_images", "node_modules", "context_files",
}

SKIP_EXTS = {
    ".pt", ".jpg", ".jpeg", ".png", ".pyc", ".so", ".pkl", ".weights",
    ".gif", ".bmp", ".ico", ".zip", ".gz", ".bin",
}

CONFIG_EXTS = {".yaml", ".yml", ".toml", ".cfg", ".txt", ".ini"}
WEB_EXTS = {".html", ".css", ".js"}

# Modules importable without being part of the project (best-effort stdlib set).
STDLIB_HINT = {
    "os", "sys", "ast", "re", "math", "json", "uuid", "enum", "gc", "io",
    "time", "typing", "pathlib", "dataclasses", "datetime", "argparse",
    "functools", "itertools", "collections", "subprocess", "logging",
    "traceback", "shutil", "tempfile", "abc", "copy", "random", "string",
    "tkinter", "unittest", "fnmatch", "tomllib", "warnings", "contextlib",
}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class FuncInfo:
    name: str
    signature: str
    docstring: str | None
    is_private: bool


@dataclass
class ClassInfo:
    name: str
    bases: list[str]
    docstring: str | None
    methods: list[FuncInfo] = field(default_factory=list)


@dataclass
class ModuleInfo:
    path: Path                       # relative to root
    module_name: str                 # dotted, relative to root
    docstring: str | None
    imports: list[str] = field(default_factory=list)   # raw top-level names
    functions: list[FuncInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    has_main: bool = False
    loc: int = 0
    parse_error: str | None = None


@dataclass
class FileInfo:
    path: Path                       # relative to root
    kind: str                        # python | config | doc | web | other
    loc: int
    purpose: str


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #

def iter_source_files(root: Path):
    """Yield relative Paths of files to include, applying exclude rules."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for name in sorted(filenames):
            ext = Path(name).suffix.lower()
            if ext in SKIP_EXTS:
                continue
            full = Path(dirpath) / name
            if full.resolve() == Path(__file__).resolve():
                continue  # don't document the generator itself
            yield full.relative_to(root)


def classify(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext == ".md":
        return "doc"
    if ext in CONFIG_EXTS:
        return "config"
    if ext in WEB_EXTS:
        return "web"
    return "other"


def count_loc(text: str) -> int:
    return text.count("\n") + (0 if text.endswith("\n") or not text else 1)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")


# --------------------------------------------------------------------------- #
# Python parsing
# --------------------------------------------------------------------------- #

def format_signature(node) -> str:
    """Render a def/async def signature using ast.unparse."""
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = "..."
    ret = ""
    if node.returns is not None:
        try:
            ret = " -> " + ast.unparse(node.returns)
        except Exception:
            ret = ""
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    return f"{prefix}{node.name}({args}){ret}"


def first_line(doc: str | None) -> str:
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def make_func(node) -> FuncInfo:
    return FuncInfo(
        name=node.name,
        signature=format_signature(node),
        docstring=ast.get_docstring(node),
        is_private=node.name.startswith("_") and not node.name.startswith("__"),
    )


def parse_python(rel_path: Path, root: Path) -> ModuleInfo:
    full = root / rel_path
    text = read_text(full)
    module_name = ".".join(rel_path.with_suffix("").parts)
    info = ModuleInfo(path=rel_path, module_name=module_name,
                      docstring=None, loc=count_loc(text))
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        info.parse_error = f"{exc.msg} (line {exc.lineno})"
        return info

    info.docstring = ast.get_docstring(tree)

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                info.imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = ("." * (node.level or 0)) + (node.module or "")
            info.imports.append(mod)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info.functions.append(make_func(node))
        elif isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                try:
                    bases.append(ast.unparse(b))
                except Exception:
                    pass
            cls = ClassInfo(name=node.name, bases=bases,
                            docstring=ast.get_docstring(node))
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cls.methods.append(make_func(sub))
            info.classes.append(cls)

    # has_main: top-level `if __name__ == "__main__":`
    for node in tree.body:
        if isinstance(node, ast.If):
            try:
                src = ast.unparse(node.test)
            except Exception:
                src = ""
            if "__name__" in src and "__main__" in src:
                info.has_main = True
                break

    return info


# --------------------------------------------------------------------------- #
# Purpose extraction (for the file map)
# --------------------------------------------------------------------------- #

def purpose_for(rel_path: Path, root: Path, kind: str,
                modules: dict[Path, ModuleInfo]) -> str:
    if kind == "python":
        mod = modules.get(rel_path)
        if mod and mod.docstring:
            return first_line(mod.docstring)
        if mod and mod.parse_error:
            return f"(parse error: {mod.parse_error})"
        return "(no module docstring)"
    text = read_text(root / rel_path)
    for line in text.splitlines():
        s = line.strip().lstrip("#").strip()
        if s:
            return s[:160]
    return "(empty)"


# --------------------------------------------------------------------------- #
# Import resolution
# --------------------------------------------------------------------------- #

def resolve_imports(modules: dict[Path, ModuleInfo]):
    """Return (internal_graph, external_set).

    internal_graph: module_name -> sorted list of internal module names it imports.
    external_set:   set of external (third-party) top-level package names.
    """
    known_basenames: dict[str, str] = {}   # short module name -> dotted name
    known_dotted = set()
    known_parts: set[str] = set()           # every package/module path component
    for m in modules.values():
        known_dotted.add(m.module_name)
        known_basenames[m.module_name.split(".")[-1]] = m.module_name
        known_parts.update(m.module_name.split("."))

    internal_graph: dict[str, list[str]] = {}
    external: set[str] = set()

    for m in modules.values():
        deps: set[str] = set()
        for imp in m.imports:
            stripped = imp.lstrip(".")
            top = stripped.split(".")[0] if stripped else ""
            last = stripped.split(".")[-1] if stripped else ""
            # relative import or matches a known project module?
            if imp.startswith("."):
                # resolve relative against package of current module
                target = known_basenames.get(last)
                if target:
                    deps.add(target)
                continue
            if stripped in known_dotted:
                deps.add(stripped)
            elif top in known_basenames:
                deps.add(known_basenames[top])
            elif top in known_parts:
                # absolute import into an internal package (e.g. `web.config`)
                if last in known_basenames:
                    deps.add(known_basenames[last])
            elif last in known_basenames:
                deps.add(known_basenames[last])
            elif top and top not in STDLIB_HINT:
                external.add(top)
        deps.discard(m.module_name)
        internal_graph[m.module_name] = sorted(deps)

    return internal_graph, external


# --------------------------------------------------------------------------- #
# Markdown helpers
# --------------------------------------------------------------------------- #

def render_tree(paths: list[Path]) -> str:
    """Render an ASCII tree from a list of relative paths."""
    tree: dict = {}
    for p in sorted(paths, key=lambda x: x.as_posix()):
        node = tree
        for part in p.parts:
            node = node.setdefault(part, {})

    lines: list[str] = ["."]

    def walk(node: dict, prefix: str):
        items = sorted(node.items(), key=lambda kv: (not kv[1], kv[0]))
        for i, (name, child) in enumerate(items):
            last = i == len(items) - 1
            connector = "└─ " if last else "├─ "
            suffix = "/" if child else ""
            lines.append(prefix + connector + name + suffix)
            if child:
                walk(child, prefix + ("   " if last else "│  "))

    walk(tree, "")
    return "\n".join(lines)


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #

def write_index(out: Path, root: Path, files: list[FileInfo],
                modules: dict[Path, ModuleInfo], stamp: str):
    n_py = sum(1 for f in files if f.kind == "python")
    n_classes = sum(len(m.classes) for m in modules.values())
    n_methods = sum(len(c.methods) for m in modules.values() for c in m.classes)
    n_funcs = sum(len(m.functions) for m in modules.values())
    total_loc = sum(f.loc for f in files)
    entry_points = sorted(m.path.as_posix() for m in modules.values() if m.has_main)

    lines = [
        "# 00 — Context Index",
        "",
        f"_Auto-generated {stamp}. Regenerate with `python generate_context.py`._",
        "",
        "TurbotProject — otolith identification: an OpenCV/PIL/YOLO desktop app "
        "(`Otolits_identyfication_program/`) plus a FastAPI web app "
        "(`Otolits_identyfication_program/web/`). Helper scripts under "
        "`Picks_modification_scripts/` and `YOLO/`.",
        "",
        "## Stats",
        "",
        f"- Files indexed: **{len(files)}** ({n_py} Python)",
        f"- Classes: **{n_classes}** · Methods: **{n_methods}** · "
        f"Top-level functions: **{n_funcs}**",
        f"- Total lines (indexed files): **{total_loc}**",
        "",
        "## Entry points (`if __name__ == \"__main__\"`)",
        "",
    ]
    if entry_points:
        lines += [f"- `{ep}`" for ep in entry_points]
    else:
        lines.append("- (none detected)")
    lines += [
        "",
        "## Quick navigation",
        "",
        "| File | What it gives you |",
        "|------|-------------------|",
        "| [01_FILE_MAP.md](01_FILE_MAP.md) | Directory tree + one-line purpose per file |",
        "| [02_CODE_MAP.md](02_CODE_MAP.md) | Classes / methods / functions, signatures + docstrings |",
        "| [03_DEPENDENCIES.md](03_DEPENDENCIES.md) | Internal import graph + external libraries |",
        "| [04_ENVIRONMENT.md](04_ENVIRONMENT.md) | requirements, configs, planning docs |",
        "",
        "## See also",
        "",
        "- `audyty_plany/kontekst_projektu.md` — hand-written deep dive "
        "(richer prose, but may lag behind the code; this index reflects the "
        "actual current source).",
        "",
    ]
    (out / "00_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def write_file_map(out: Path, root: Path, files: list[FileInfo]):
    lines = [
        "# 01 — File Map",
        "",
        "## Directory tree",
        "",
        "```",
        render_tree([f.path for f in files]),
        "```",
        "",
        "## Files",
        "",
        "| Path | Type | LOC | Purpose |",
        "|------|------|-----|---------|",
    ]
    for f in sorted(files, key=lambda x: x.path.as_posix()):
        lines.append(
            f"| `{f.path.as_posix()}` | {f.kind} | {f.loc} | "
            f"{md_escape(f.purpose)} |"
        )
    lines.append("")
    (out / "01_FILE_MAP.md").write_text("\n".join(lines), encoding="utf-8")


def write_code_map(out: Path, modules: dict[Path, ModuleInfo]):
    lines = ["# 02 — Code Map", "",
             "All Python modules with classes, methods and functions "
             "(including `_private`), signatures and docstrings.", ""]
    for path in sorted(modules, key=lambda p: p.as_posix()):
        m = modules[path]
        lines.append(f"## `{path.as_posix()}`")
        lines.append("")
        if m.parse_error:
            lines.append(f"> ⚠ parse error: {m.parse_error}")
            lines.append("")
            continue
        if m.docstring:
            lines.append(f"_{first_line(m.docstring)}_")
            lines.append("")
        if m.has_main:
            lines.append("**Has `__main__` entry point.**")
            lines.append("")

        for fn in m.functions:
            lines.append(f"- `{fn.signature}`")
            if fn.docstring:
                lines.append(f"  - {first_line(fn.docstring)}")
        if m.functions:
            lines.append("")

        for cls in m.classes:
            bases = f"({', '.join(cls.bases)})" if cls.bases else ""
            lines.append(f"### class `{cls.name}{bases}`")
            if cls.docstring:
                lines.append(f"_{first_line(cls.docstring)}_")
            lines.append("")
            for meth in cls.methods:
                lines.append(f"- `{meth.signature}`")
                if meth.docstring:
                    lines.append(f"  - {first_line(meth.docstring)}")
            lines.append("")
        if not m.functions and not m.classes and not m.docstring:
            lines.append("_(no top-level classes/functions)_")
            lines.append("")
    (out / "02_CODE_MAP.md").write_text("\n".join(lines), encoding="utf-8")


def write_dependencies(out: Path, modules: dict[Path, ModuleInfo]):
    internal_graph, external = resolve_imports(modules)
    lines = ["# 03 — Dependencies", "",
             "## Internal import graph", "",
             "Each module and the project modules it imports.", "",
             "| Module | Imports (internal) |",
             "|--------|--------------------|"]
    for mod in sorted(internal_graph):
        deps = internal_graph[mod]
        dep_str = ", ".join(f"`{d}`" for d in deps) if deps else "—"
        lines.append(f"| `{mod}` | {dep_str} |")

    lines += ["", "## External libraries", "",
              "Third-party top-level packages imported across the project:", ""]
    if external:
        lines += [f"- `{name}`" for name in sorted(external)]
    else:
        lines.append("- (none detected)")

    lines += ["", "## Entry points and their fan-out", ""]
    eps = [m for m in modules.values() if m.has_main]
    if eps:
        for m in sorted(eps, key=lambda x: x.module_name):
            deps = internal_graph.get(m.module_name, [])
            dep_str = ", ".join(f"`{d}`" for d in deps) if deps else "—"
            lines.append(f"- `{m.path.as_posix()}` → {dep_str}")
    else:
        lines.append("- (none detected)")
    lines.append("")
    (out / "03_DEPENDENCIES.md").write_text("\n".join(lines), encoding="utf-8")


def write_environment(out: Path, root: Path, files: list[FileInfo]):
    lines = ["# 04 — Environment & Non-code Context", ""]

    def dump_file(rel: Path, header: str, lang: str = "text", limit: int = 120):
        full = root / rel
        if not full.exists():
            return
        lines.append(f"## {header}")
        lines.append(f"`{rel.as_posix()}`")
        lines.append("")
        text = read_text(full)
        body = text.splitlines()
        truncated = len(body) > limit
        snippet = "\n".join(body[:limit])
        lines.append(f"```{lang}")
        lines.append(snippet)
        if truncated:
            lines.append(f"... ({len(body) - limit} more lines)")
        lines.append("```")
        lines.append("")

    # requirements files
    for f in files:
        if f.path.name == "requirements.txt":
            dump_file(f.path, f"Requirements — {f.path.parent.as_posix() or '.'}")

    dump_file(Path("pyvenv.cfg"), "pyvenv.cfg", "ini")

    # YAML/config files
    for f in sorted(files, key=lambda x: x.path.as_posix()):
        if f.path.suffix.lower() in {".yaml", ".yml"}:
            dump_file(f.path, f"Config — {f.path.as_posix()}", "yaml")

    # planning / audit docs inventory
    docs = [f for f in files if f.kind == "doc"
            and "audyty_plany" in f.path.parts]
    if docs:
        lines.append("## Planning / audit documents (`audyty_plany/`)")
        lines.append("")
        lines.append("| Document | Top heading |")
        lines.append("|----------|-------------|")
        for d in sorted(docs, key=lambda x: x.path.as_posix()):
            lines.append(f"| `{d.path.as_posix()}` | {md_escape(d.purpose)} |")
        lines.append("")

    (out / "04_ENVIRONMENT.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Project root to scan (default: script directory).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: <root>/context_files).")
    args = parser.parse_args()

    root = args.root.resolve()
    out = (args.out or (root / "context_files")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    rel_files = list(iter_source_files(root))

    modules: dict[Path, ModuleInfo] = {}
    for rel in rel_files:
        if classify(rel) == "python":
            modules[rel] = parse_python(rel, root)

    files: list[FileInfo] = []
    for rel in rel_files:
        kind = classify(rel)
        loc = modules[rel].loc if rel in modules else count_loc(read_text(root / rel))
        files.append(FileInfo(path=rel, kind=kind, loc=loc,
                              purpose=purpose_for(rel, root, kind, modules)))

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    write_index(out, root, files, modules, stamp)
    write_file_map(out, root, files)
    write_code_map(out, modules)
    write_dependencies(out, modules)
    write_environment(out, root, files)

    print(f"Context written to {out}")
    print(f"  files indexed : {len(files)} ({len(modules)} python)")
    print(f"  classes       : {sum(len(m.classes) for m in modules.values())}")
    print(f"  methods       : {sum(len(c.methods) for m in modules.values() for c in m.classes)}")
    print(f"  functions     : {sum(len(m.functions) for m in modules.values())}")
    print("  outputs       : 00_INDEX.md 01_FILE_MAP.md 02_CODE_MAP.md "
          "03_DEPENDENCIES.md 04_ENVIRONMENT.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""生成 MODULE_DOCS/99-symbol-index/ 下的符号索引。

用法（在仓库根目录）：
    python3 MODULE_DOCS/tools/gen_symbol_index.py

做什么：ast 解析 python/freetoken 下的每个 .py（不 import、不执行），按顶层目录分桶，
逐文件输出「行号 | 符号 | 说明」表格。符号包含模块级函数、类、类方法，以及**嵌套函数**
（用 ↳ 标出层级）。说明列优先取 docstring 首行，没有 docstring 时回退到签名的首行。

设计约束（改代码时请保持）：
  * 纯 ast，不执行任何代码 —— 在无 torch/GPU 的机器上也能跑；
  * 行号是该符号在源码里的 def/class 起始行；
  * "行数"是逻辑行数（末行无换行时算 1 行），可能与 wc -l 差 1。
"""
from __future__ import annotations

import ast
import os
import re
import sys
from collections import defaultdict

ROOT = "python/freetoken"
OUT = "MODULE_DOCS/99-symbol-index"

DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def first_doc_line(node: ast.AST) -> str:
    doc = ast.get_docstring(node)
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        line = line.strip()
        if line:
            return re.sub(r"\s+", " ", line)
    return ""


def signature(node: ast.AST, src: str) -> str:
    """def/class 声明到 ':' 为止的第一段（多行签名会被截断）。"""
    seg = ast.get_source_segment(src, node) or ""
    head = []
    for line in seg.splitlines():
        head.append(line.strip())
        if line.rstrip().endswith(":"):
            break
    text = " ".join(head)
    text = text.split(":", 1)[0] if ":" in text else text
    text = re.sub(r"^(async\s+)?def\s+", "", text)
    return text if len(text) <= 150 else text[:147] + "..."


def module_of(path: str) -> str:
    rel = os.path.relpath(path, ROOT)
    parts = rel.split(os.sep)
    return "(top-level)" if len(parts) == 1 else parts[0]


def collect(tree: ast.AST, src: str):
    """返回 [(depth, kind, name, lineno, desc)]，递归进函数体。"""
    rows = []

    def visit_body(body, depth, kind_ctx):
        for node in body:
            if isinstance(node, ast.ClassDef):
                desc = first_doc_line(node) or signature(node, src)
                rows.append((depth, "class", node.name, node.lineno, desc))
                visit_body(node.body, depth + 1, "class")
            elif isinstance(node, DEF_NODES):
                desc = first_doc_line(node) or signature(node, src)
                rows.append((depth, kind_ctx if depth else "function", node.name, node.lineno, desc))
                visit_body(node.body, depth + 1, "nested")
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                # 模块级/类体内的条件定义（如 if TYPE_CHECKING:）也要收
                for field in ("body", "orelse", "finalbody"):
                    sub = getattr(node, field, None)
                    if sub:
                        visit_body(sub, depth, kind_ctx)
                for h in getattr(node, "handlers", []) or []:
                    visit_body(h.body, depth, kind_ctx)
            elif hasattr(ast, "Match") and isinstance(node, ast.Match):
                # match/case 分支里也可能定义函数（rotary.py 的 yarn 分支就是）
                for case in node.cases:
                    visit_body(case.body, depth, kind_ctx)

    visit_body(tree.body, 0, "function")
    return rows


def main() -> int:
    buckets = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            src = open(path, encoding="utf-8").read()
            lines = src.count("\n") + (0 if src.endswith("\n") else 1)
            try:
                tree = ast.parse(src)
            except SyntaxError as exc:
                print(f"skip unparsable {path}: {exc}", file=sys.stderr)
                continue
            buckets[module_of(path)].append(
                (os.path.relpath(path, "python").replace(os.sep, "/"), lines, collect(tree, src))
            )

    os.makedirs(OUT, exist_ok=True)
    grand = 0
    for module, files in sorted(buckets.items()):
        safe = module.strip("()").replace("/", "_").replace(" ", "_")
        out = os.path.join(OUT, f"{safe}.md")
        n_sym = sum(len(rows) for _, _, rows in files)
        n_loc = sum(lines for _, lines, _ in files)
        grand += n_sym
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(f"# 符号索引：`{module}`\n\n")
            fh.write(
                f"自动生成（`ast` 解析，不执行代码）。**{len(files)} 个文件 / {n_loc} 行 / {n_sym} 个符号**。\n\n"
            )
            fh.write("> 用途：确认某个模块里**到底有哪些**类/函数/方法及其行号，避免遗漏。\n")
            fh.write("> 行为详解请看同目录下的手写模块文档。\n")
            fh.write("> 重新生成：`python3 MODULE_DOCS/tools/gen_symbol_index.py`\n\n")
            fh.write("| 文件 | 行数 | 符号数 |\n|---|---:|---:|\n")
            for rel, lines, rows in files:
                fh.write(f"| `{rel}` | {lines} | {len(rows)} |\n")
            fh.write("\n---\n\n")
            for rel, lines, rows in files:
                fh.write(f"## `{rel}`  ({lines} 行)\n\n")
                if not rows:
                    fh.write("_（无类/函数定义，模块级代码）_\n\n")
                    continue
                fh.write("| 行 | 符号 | 说明 |\n|---:|---|---|\n")
                for depth, kind, name, lineno, desc in rows:
                    if kind == "class":
                        label = f"**class `{name}`**"
                    elif depth == 0:
                        label = f"`{name}()`"
                    else:
                        label = "&nbsp;" * (3 * depth) + f"↳ `{name}()`"
                    desc = desc.replace("|", "\\|")
                    if len(desc) > 150:
                        desc = desc[:147] + "..."
                    fh.write(f"| {lineno} | {label} | {desc} |\n")
                fh.write("\n")
        print(f"{out}: {len(files)} files / {n_sym} symbols")
    print(f"TOTAL: {grand} symbols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

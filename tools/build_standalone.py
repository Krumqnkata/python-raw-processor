"""Assemble a single-script distribution from the canonical modular sources."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def without_local_imports(source: str) -> str:
    lines = source.splitlines(keepends=True)
    remove = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module in {
            '__future__', 'raw_engine', 'batch', 'app_gui',
        }:
            remove.update(range(node.lineno - 1, node.end_lineno))
    return ''.join(line for index, line in enumerate(lines) if index not in remove)


def build():
    parts = [
        '"""RAW Studio: complete standalone application. Generated from the modular project."""\n',
        'from __future__ import annotations\n',
    ]
    for name in ['raw_engine.py', 'batch.py', 'app_gui.py', 'raw_processor.py']:
        parts.append(f'\n# ---------- {name} ----------\n')
        parts.append(without_local_imports((ROOT / name).read_text(encoding='utf-8')))
    target = ROOT / 'RAW_Studio.py'
    target.write_text(''.join(parts), encoding='utf-8')
    print(target)
    return target


if __name__ == '__main__':
    build()

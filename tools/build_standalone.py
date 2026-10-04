"""Assemble a single-script distribution from the canonical modular sources."""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def without_local_imports(source: str) -> str:
    lines = source.splitlines(keepends=True)
    remove = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module in {
            '__future__', 'raw_engine', 'batch', 'app_gui', 'imaging', 'metadata', 'studio', 'studio_workflow', 'experience', 'packaging_smoke',
        }:
            remove.update(range(node.lineno - 1, node.end_lineno))
    return ''.join(line for index, line in enumerate(lines) if index not in remove)


def build(check=False):
    parts = [
        '"""RAW Studio: complete standalone application. Generated from the modular project."""\n',
        'from __future__ import annotations\n',
    ]
    for name in ['imaging.py', 'metadata.py', 'raw_engine.py', 'studio.py', 'batch.py', 'studio_workflow.py', 'experience.py', 'app_gui.py', 'packaging_smoke.py', 'raw_processor.py']:
        parts.append(f'\n# ---------- {name} ----------\n')
        parts.append(without_local_imports((ROOT / name).read_text(encoding='utf-8')))
    target = ROOT / 'RAW_Studio.py'
    source = ''.join(parts)
    if check:
        if not target.is_file() or target.read_text(encoding='utf-8') != source:
            raise SystemExit('RAW_Studio.py is stale. Run python tools/build_standalone.py.')
        print('Standalone sources are in sync.')
    else:
        target.write_text(source, encoding='utf-8')
        print(target)
    return target


if __name__ == '__main__':
    build(check='--check' in sys.argv)

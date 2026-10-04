"""Run each GUI test in a fresh Python/Tk process, on Windows and Linux.

Repeated Tcl interpreter creation/teardown can interfere with later Tk roots
on Windows. The application has one root per process; test that same lifecycle
instead of sharing native Tk state between unrelated tests.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    collected = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q', 'tests/test_gui.py'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if collected.returncode:
        print(collected.stdout, end='')
        print(collected.stderr, end='', file=sys.stderr)
        return collected.returncode
    nodes = [line.strip() for line in collected.stdout.splitlines()
             if line.startswith('tests/test_gui.py::')]
    if not nodes:
        print('No GUI tests collected.', file=sys.stderr)
        return 1
    failed = 0
    for node in nodes:
        print(f'Checking {node}', flush=True)
        result = subprocess.run([sys.executable, '-m', 'pytest', '-q', node], cwd=ROOT)
        if result.returncode:
            failed += 1
    print(f'GUI checks: {len(nodes) - failed}/{len(nodes)} passed.', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())

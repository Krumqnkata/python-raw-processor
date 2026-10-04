"""Build a portable Windows app. Inno Setup wraps the same directory in an installer."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    if sys.platform != 'win32': raise SystemExit('Build Windows packages on Windows.')
    subprocess.run([sys.executable,'-m','PyInstaller','--noconfirm','--clean','--windowed',
                    '--name','RAWStudio','--collect-all','customtkinter','--collect-all','tkinterdnd2',
                    '--collect-all','rawpy',str(ROOT/'raw_processor.py')],cwd=ROOT,check=True)

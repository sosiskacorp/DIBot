"""Точка входа для деплоя в Streamlit Community Cloud (share.streamlit.io)."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

runpy.run_path(str(SRC / "app.py"), run_name="__main__")

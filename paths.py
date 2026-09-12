"""
Repo-root-relative data paths.

Modules previously hardcoded "../data/…", which only resolves when the script
is run from inside its own package directory. `python features/momentum_features.py`
from the repo root failed, while `cd features && python momentum_features.py`
worked — a difference nothing in the README mentioned. Anchoring on this file's
location makes every entry point behave the same from anywhere.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

RAW_DATA_PATH = DATA / "raw_metals_data.csv"
FEATURE_PATH = DATA / "features.csv"
MACRO_PATH = DATA / "macro_data.csv"
BACKTEST_RESULTS_PATH = DATA / "backtest_results.csv"

DATA.mkdir(exist_ok=True)

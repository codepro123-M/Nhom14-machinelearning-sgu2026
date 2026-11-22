# -*- coding: utf-8 -*-
"""Hằng số dùng chung cho toàn bộ dự án (theo Cẩm nang, giai đoạn 0.4)."""
from pathlib import Path

SEED = 42
TEST_SIZE = 0.2
N_SPLITS = 5
TARGET = "y"

# Thư mục gốc của dự án (thư mục cha của src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "bank-additional-full.csv"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Ensure tests use the lightweight FAISS stub to avoid binary import crashes on unsupported systems.
os.environ.setdefault("BTC_USE_FAISS_STUB", "1")

#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from scanpdf.preprocess import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compute Paper-1 headline CIs, significance, and LaTeX tables (post-hoc only)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fedpbg.reporting.headline_statistics import run_all  # noqa: E402

if __name__ == "__main__":
    run_all()

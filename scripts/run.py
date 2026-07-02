"""ShadowInfer CLI entry-point script.

This is a thin wrapper around ``shadowinfer.__main__`` so that the CLI can be
run directly from the repository root without installing the package:

    python scripts/run.py profiler --model Fast-dLLM-v2-7B --config configs/profiler_full.yaml
    python scripts/run.py optimize --model Fast-dLLM-v2-7B --config configs/optimize_full.yaml
    python scripts/run.py serve --model Fast-dLLM-v2-7B --config configs/optimize_full.yaml

对应文档：ARCHITECTURE.md §4.2 端到端推理流程
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running scripts/run.py directly from the repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shadowinfer.__main__ import main

if __name__ == "__main__":
    main()

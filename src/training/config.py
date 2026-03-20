from __future__ import annotations

from src.targets.splitter import SplitConfig

PRODUCTION_SPLIT_CONFIG = SplitConfig(
    min_train_rows=756,
    test_rows=63,
    step_rows=21,
    gap_rows=2,
    embargo_rows=5,
)

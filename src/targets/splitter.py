"""
src/targets/splitter.py
========================
Task 13 — Walk-forward time-series cross-validation splitter.

Design
------
*  Expanding-window (anchored) splits — the train window grows with each fold.
*  Configurable gap between train end and test start (default = 1 day).
*  Configurable minimum train size and test size (in calendar days or rows).
*  Returns a list of (train_idx, test_idx) index arrays for use with any
   sklearn-compatible estimator.
*  Also provides a ``PurgedKFold``-style helper that removes rows within
   ``embargo_days`` of the test boundary from the train set to prevent
   information leakage across the border.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SplitConfig:
    """Parameters that govern the walk-forward split."""

    # Minimum number of *rows* in the initial training window.
    min_train_rows: int = 504        # ~2 years of daily data

    # Number of *rows* in each test fold.
    test_rows: int = 63              # ~1 quarter

    # Number of *rows* to advance the window each fold (step).
    step_rows: int = 63              # non-overlapping test windows

    # Gap rows between train end and test start (leakage buffer).
    gap_rows: int = 1

    # Additional embargo rows removed from the *end* of the train set.
    embargo_rows: int = 0

    # If True: expanding window (grows).  If False: rolling window (fixed size).
    expanding: bool = True

    # Maximum train size for rolling window mode (ignored if expanding=True).
    max_train_rows: Optional[int] = None


# ---------------------------------------------------------------------------
# Core splitter
# ---------------------------------------------------------------------------

class WalkForwardSplitter:
    """
    Scikit-learn compatible walk-forward splitter.

    Parameters
    ----------
    config : SplitConfig instance.

    Example
    -------
    >>> wf = WalkForwardSplitter()
    >>> splits = wf.split(X)
    >>> for fold_idx, (tr, te) in enumerate(splits):
    ...     X_train, y_train = X.iloc[tr], y.iloc[tr]
    ...     X_test,  y_test  = X.iloc[te], y.iloc[te]
    """

    def __init__(self, config: Optional[SplitConfig] = None):
        self.cfg = config or SplitConfig()

    # ------------------------------------------------------------------
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,   # unused, kept for sklearn compat
        groups: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward folds.

        Returns
        -------
        List of (train_indices, test_indices) as integer position arrays.
        """
        n = len(X)
        cfg = self.cfg

        if n < cfg.min_train_rows + cfg.gap_rows + cfg.test_rows:
            raise ValueError(
                f"Dataset too small ({n} rows) for min_train={cfg.min_train_rows}, "
                f"gap={cfg.gap_rows}, test={cfg.test_rows}."
            )

        splits = []
        train_end = cfg.min_train_rows   # exclusive upper bound of train

        while True:
            test_start = train_end + cfg.gap_rows
            test_end   = test_start + cfg.test_rows

            if test_end > n:
                break   # not enough data for a full test fold

            # Build train indices (with optional embargo)
            embargo_end = train_end - cfg.embargo_rows
            if cfg.expanding:
                train_start = 0
            else:
                max_tr = cfg.max_train_rows or cfg.min_train_rows
                train_start = max(0, embargo_end - max_tr)

            train_idx = np.arange(train_start, embargo_end)
            test_idx  = np.arange(test_start,  test_end)

            splits.append((train_idx, test_idx))

            train_end += cfg.step_rows

        logger.info("WalkForward: %d folds (min_train=%d, test=%d, step=%d, gap=%d)",
                    len(splits), cfg.min_train_rows, cfg.test_rows,
                    cfg.step_rows, cfg.gap_rows)
        return splits

    # ------------------------------------------------------------------
    def get_n_splits(self, X: Optional[pd.DataFrame] = None, **kw) -> int:
        """Return number of folds (requires X)."""
        if X is None:
            raise ValueError("X required to compute n_splits.")
        return len(self.split(X))

    # ------------------------------------------------------------------
    def split_dates(
        self, X: pd.DataFrame
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Same as split() but returns DatetimeIndex pairs."""
        return [
            (X.index[tr], X.index[te])
            for tr, te in self.split(X)
        ]

    # ------------------------------------------------------------------
    def describe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a summary DataFrame of all folds (useful for inspection)."""
        rows = []
        for i, (tr, te) in enumerate(self.split(X)):
            rows.append({
                "fold":       i,
                "train_start": X.index[tr[0]],
                "train_end":   X.index[tr[-1]],
                "test_start":  X.index[te[0]],
                "test_end":    X.index[te[-1]],
                "train_rows":  len(tr),
                "test_rows":   len(te),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience: single final hold-out split
# ---------------------------------------------------------------------------

def train_test_final_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_frac: float = 0.15,
    gap_rows: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simple chronological train / hold-out split.

    Parameters
    ----------
    test_frac  : Fraction of rows to hold out at the end.
    gap_rows   : Rows to exclude between train and test.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    n          = len(X)
    test_start = int(n * (1 - test_frac)) + gap_rows
    train_end  = test_start - gap_rows

    X_train, X_test = X.iloc[:train_end], X.iloc[test_start:]
    y_train, y_test = y.iloc[:train_end], y.iloc[test_start:]

    logger.info(
        "Final split: train %s–%s (%d rows)  test %s–%s (%d rows)",
        X_train.index[0].date(), X_train.index[-1].date(), len(X_train),
        X_test.index[0].date(),  X_test.index[-1].date(),  len(X_test),
    )
    return X_train, X_test, y_train, y_test

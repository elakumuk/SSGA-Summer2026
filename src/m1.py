"""M1 — the SIMPLE, STATIC, LINEAR layer.

Score = fixed-ratio linear blend of TECHNICAL (momentum+trend) and MACRO tilt.
No learning, no dynamics, no regime adaptation here -- all of that lives in M2.
This keeps "valuation margin" (headroom) for the M2 layer on top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import M1Config


class M1Model:
    def __init__(self, cfg: M1Config) -> None:
        self.cfg = cfg

    def score(self, factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Fixed-ratio linear combination of factor GROUPS -> one M1 score per
        (date, ticker). Each factor frame is already cross-sectionally z-scored,
        so they are comparable. Weights come from cfg.factors (technical/risk/macro)."""
        w = self.cfg.factors
        ref = next(iter(factors.values()))
        total = pd.DataFrame(0.0, index=ref.index, columns=ref.columns)
        for name, frame in factors.items():
            total = total + w.get(name, 0.0) * frame.reindex_like(ref).fillna(0.0)
        return total

    def active_tilts(self, score: pd.DataFrame, max_tilt: float, max_abs_weight: float) -> pd.DataFrame:
        """Convert the cross-sectional M1 score into BENCHMARK-RELATIVE active weights.

        benchmark = equal weight (1/N). Each week, the highest-scored assets get a
        positive tilt and the lowest get a negative tilt (synthetic underweight), so
        no outright shorting is needed. Weights are clipped and renormalized to be
        fully invested. This is the 'how we allocate the 3 assets' demonstration.
        """
        out = pd.DataFrame(0.0, index=score.index, columns=score.columns)
        for date, row in score.iterrows():
            s = row.dropna()
            if s.empty:
                continue
            n = len(s)
            base = 1.0 / n
            # rank in [-1, +1]: best name +1, worst -1
            ranks = s.rank()
            centered = 2 * (ranks - 1) / (n - 1) - 1 if n > 1 else pd.Series(0.0, index=s.index)
            tilt = max_tilt * centered
            w = (base + tilt).clip(lower=0.0, upper=max_abs_weight)
            if w.sum() > 0:
                w = w / w.sum()  # fully invested, gross = 1
            out.loc[date, w.index] = w.values
        return out

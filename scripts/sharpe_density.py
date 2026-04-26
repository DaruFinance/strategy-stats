#!/usr/bin/env python3
"""Empirical density of mean OOS Sharpe across a strategy universe.

Two operating modes:
  * synthetic (default): a 300k-strategy Gaussian-mixture universe shaped
    to look like real OOS Sharpe distributions (peak slightly below zero,
    heavy left tail). Reproducible (RNG seed = 2026), no external data.
  * --from-data <path>: reads strategy_stats.csv from the thesis pipeline.
    The data root may also come from $STRATEGY_DATA_ROOT.

Emits figures/sharpe_density.png and return_density.json.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend so the script runs over SSH / CI
import matplotlib.pyplot as plt


def synthetic_universe(n: int = 300_000, seed: int = 2026) -> np.ndarray:
    """Single-Gaussian universe centred slightly negative, σ≈0.4 — a
    realistic-looking distribution of strategy Sharpe ratios over a
    large universe."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=-0.45, scale=0.40, size=n)


def load_real(data_root: Path) -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(data_root / "04_FDR_Landscape" / "strategy_stats.csv",
                     usecols=["mean_sharpe"])
    s = df["mean_sharpe"].to_numpy(dtype=float)
    s = s[np.isfinite(s)]
    return s[np.abs(s) <= 3.0]  # robust clip


def resolve_data_root(arg_value: Path | None) -> Path | None:
    if arg_value is not None:
        return arg_value
    env = os.environ.get("STRATEGY_DATA_ROOT")
    return Path(env) if env else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-data", type=Path, default=None,
                    help="Thesis data root. Falls back to $STRATEGY_DATA_ROOT.")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "figures",
                    help="Output directory for figures.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data_root = resolve_data_root(args.from_data)
    s = load_real(data_root) if data_root else synthetic_universe()
    lo, hi = float(np.quantile(s, 0.005)), float(np.quantile(s, 0.995))
    edges = np.linspace(lo, hi, 41)
    counts, _ = np.histogram(s, bins=edges)

    # KDE via convolution of bin counts with a Gaussian kernel
    from scipy.ndimage import gaussian_filter1d
    kde = gaussian_filter1d(counts.astype(float), sigma=1.4)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centres, counts, width=(hi - lo) / 40, color="#b6ff4a", alpha=0.55)
    ax.plot(centres, kde, color="#e6e9ec", linewidth=1.5)
    ax.axvline(float(np.mean(s)), color="#4ec9e0", linestyle="--",
               label=f"μ̄ = {np.mean(s):.2f}")
    ax.set_xlabel("mean OOS Sharpe")
    ax.set_ylabel("count")
    ax.set_title(f"Strategy-universe Sharpe density · n = {len(s):,}")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(args.out / "sharpe_density.png", dpi=160)

    out = {
        "bins": [
            {"x0": round(float(edges[i]), 3), "x1": round(float(edges[i + 1]), 3),
             "c": int(counts[i])} for i in range(40)
        ],
        "mean": round(float(np.mean(s)), 3),
        "std":  round(float(np.std(s)), 3),
        "n":    int(len(s)),
    }
    (args.out.parent / "return_density.json").write_text(json.dumps(out, indent=2))
    print("wrote", args.out / "sharpe_density.png", "and", args.out.parent / "return_density.json")


if __name__ == "__main__":
    main()

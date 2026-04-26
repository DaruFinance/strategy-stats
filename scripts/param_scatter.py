#!/usr/bin/env python3
"""(μ, σ, t) parameter-space scatter across a strategy universe.

Two operating modes:
  * synthetic (default): a per-asset 12-strategy stratified sample
    drawn from class-typical Gaussian moments. Reproducible (RNG seed = 2026).
  * --from-data <path>: reads the thesis pipeline's strategy_stats.csv,
    stratified subsample of 12 strategies per asset. The data root may also
    come from $STRATEGY_DATA_ROOT.

Emits figures/param_scatter.png and param_scatter.json.
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


ASSETS = ["BTC", "DOGE", "SOL", "BNB", "EURUSD", "USDJPY", "EURGBP", "XAUUSD", "WTI"]
CLASSES = {
    "BTC": "Crypto", "DOGE": "Crypto", "SOL": "Crypto", "BNB": "Crypto",
    "EURUSD": "FX", "USDJPY": "FX", "EURGBP": "FX",
    "XAUUSD": "Commodities", "WTI": "Commodities",
}
CLASS_COLOUR = {"Crypto": "#b6ff4a", "FX": "#4ec9e0", "Commodities": "#e0a04a"}


def synthetic(n_per_asset: int = 12, seed: int = 2026) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for a in ASSETS:
        mus = rng.normal(-0.4, 0.35, size=n_per_asset)
        sgs = np.abs(rng.normal(1.0, 0.3, size=n_per_asset))
        tts = mus / (sgs / np.sqrt(20))
        for mu, sg, t in zip(mus, sgs, tts):
            rows.append({"mu": float(mu), "sg": float(sg), "t": float(t), "asset": a})
    return rows


def from_real(data_root: Path) -> list[dict]:
    import pandas as pd
    df = pd.read_csv(data_root / "04_FDR_Landscape" / "strategy_stats.csv",
                     usecols=["mean_sharpe", "std_sharpe", "t_stat", "asset"])
    df = df[df["asset"].isin(ASSETS)]
    df = df[df["mean_sharpe"].abs() <= 3.0].copy()
    df = df[df["std_sharpe"].abs() <= 3.0].copy()
    # per-asset stratified subsample
    rng = np.random.default_rng(20260424)
    picks = []
    for a, sub in df.groupby("asset"):
        picks.append(sub.sample(n=min(12, len(sub)), random_state=int(rng.integers(1e9))))
    samp = __import__("pandas").concat(picks, ignore_index=True)
    rows = [
        {"mu": float(r.mean_sharpe), "sg": float(r.std_sharpe),
         "t": float(r.t_stat), "asset": r.asset}
        for r in samp.itertuples(index=False)
    ]
    return rows


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
    points = from_real(data_root) if data_root else synthetic()

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for a in ASSETS:
        xs = [p["mu"] for p in points if p["asset"] == a]
        ys = [p["sg"] for p in points if p["asset"] == a]
        ax.scatter(xs, ys, s=34, color=CLASS_COLOUR[CLASSES[a]], alpha=0.85,
                   edgecolors="#101820", linewidths=0.4, label=a)
    ax.axvline(0, color="#6f7680", linewidth=0.6, linestyle="--")
    ax.set_xlabel("mean OOS Sharpe (μ)")
    ax.set_ylabel("Sharpe dispersion (σ)")
    ax.set_title("Per-strategy (μ, σ) scatter")
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(args.out / "param_scatter.png", dpi=160)

    (args.out.parent / "param_scatter.json").write_text(
        json.dumps({"points": points, "n_total": len(points)}, indent=2))
    print("wrote", args.out / "param_scatter.png", "and", args.out.parent / "param_scatter.json")


if __name__ == "__main__":
    main()

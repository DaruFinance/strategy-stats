# strategy-stats

**Population-level statistics over a universe of algorithmic trading strategies.**

> Utility layer for [Sharpe density and population scatter](https://daru.finance/projects/strategy-stats) and the rest of the Lab Daniel Gatto publishes on [daru.finance](https://daru.finance).

Two reproducible pipelines over a large strategy population:

1. **Sharpe-ratio density** — kernel-density estimate of the mean
   out-of-sample Sharpe across the universe, after robust clipping.
2. **(μ, σ, t) parameter-space scatter** — per-strategy Sharpe mean,
   Sharpe standard deviation, and aggregated t-statistic, stratified by
   asset and asset-class.

## Reproduce

```bash
git clone https://github.com/DaruFinance/strategy-stats
cd strategy-stats
pip install -e .
python scripts/sharpe_density.py
python scripts/param_scatter.py
```

Both scripts run a deterministic synthetic universe and write
`figures/sharpe_density.png` + `return_density.json` and
`figures/param_scatter.png` + `param_scatter.json`. No external data needed.

## Problem statement

Given `N` strategies, each evaluated on `T_i` walk-forward windows,
define `μ_i = mean OOS Sharpe`, `σ_i = std OOS Sharpe`,
`t_i = μ_i √T_i / σ_i`. The joint distribution of `(μ, σ, t)` across
strategies is the object of interest; marginal distributions of `μ` (and
per-asset medians) indicate the population's first-order behaviour.

## Usage

The default mode runs a synthetic universe (Gaussian mixture, RNG seed = 2026).

To reproduce the per-asset thesis figures, point either script at the data root
either via `--from-data` or the `STRATEGY_DATA_ROOT` environment variable:

```bash
export STRATEGY_DATA_ROOT="$HOME/PhD_Research"   # adjust for your machine
python scripts/sharpe_density.py
python scripts/param_scatter.py
```

The expected layout under `$STRATEGY_DATA_ROOT` is:

```
04_FDR_Landscape/strategy_stats.csv
```

Strategy identifiers are stripped before serialisation: only
`(mean_sharpe, std_sharpe, t_stat, asset, asset_class)` rows are retained.

## Output

- `figures/sharpe_density.png` + `return_density.json`
- `figures/param_scatter.png` + `param_scatter.json`

## License

MIT © Daniel Vieira Gatto.

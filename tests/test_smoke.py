"""Smoke test: both synthetic demos run end-to-end."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SHARPE = REPO_ROOT / "scripts" / "sharpe_density.py"
PARAM = REPO_ROOT / "scripts" / "param_scatter.py"


def _run(script: Path, out_dir: Path) -> None:
    res = subprocess.run(
        [sys.executable, str(script), "--out", str(out_dir)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert res.returncode == 0, res.stderr


def test_sharpe_density_demo(tmp_path: Path) -> None:
    out = tmp_path / "figs"
    _run(SHARPE, out)
    fig = out / "sharpe_density.png"
    assert fig.exists() and fig.stat().st_size > 5_000
    payload = json.loads((out.parent / "return_density.json").read_text())
    assert payload["n"] > 0 and len(payload["bins"]) > 0


def test_param_scatter_demo(tmp_path: Path) -> None:
    out = tmp_path / "figs"
    _run(PARAM, out)
    fig = out / "param_scatter.png"
    assert fig.exists() and fig.stat().st_size > 5_000
    payload = json.loads((out.parent / "param_scatter.json").read_text())
    assert payload["n_total"] > 0 and len(payload["points"]) == payload["n_total"]


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_sharpe_density_demo(Path(d) / "a")
        test_param_scatter_demo(Path(d) / "b")
        print("ok")

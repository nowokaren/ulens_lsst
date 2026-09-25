"""Small, non-production analyses for controlled validation products."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .scientific_validation import summarize_validation


def write_validation_summary(rows: Iterable[Mapping], path: str | Path) -> dict:
    """Write descriptive validation statistics as JSON and return them."""

    summary = summarize_validation(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def plot_validation_diagnostics(rows: Iterable[Mapping], output_dir: str | Path) -> None:
    """Create flux-residual, pull, and position diagnostic plots."""

    import matplotlib.pyplot as plt

    rows = list(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = np.asarray([r["expected_inst_flux"] for r in rows], dtype=float)
    residual = np.asarray([r["fractional_flux_residual"] for r in rows], dtype=float)
    magnitude = np.asarray([r["requested_mag"] for r in rows], dtype=float)
    pulls = np.asarray(
        [r["pull"] for r in rows if r.get("pull") is not None], dtype=float
    )

    figure, axis = plt.subplots()
    axis.scatter(expected, residual)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Expected instrumental flux")
    axis.set_ylabel("Fractional baseline-subtracted flux residual")
    figure.savefig(output_dir / "fractional_flux_residual_vs_expected_flux.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.scatter(magnitude, residual)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Requested magnitude")
    axis.set_ylabel("Fractional baseline-subtracted flux residual")
    figure.savefig(output_dir / "fractional_flux_residual_vs_magnitude.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots()
    if pulls.size:
        axis.hist(pulls, bins=min(20, max(3, pulls.size)))
    axis.set_xlabel("Flux pull")
    axis.set_ylabel("Count")
    figure.savefig(output_dir / "flux_pull_distribution.png", dpi=140)
    plt.close(figure)

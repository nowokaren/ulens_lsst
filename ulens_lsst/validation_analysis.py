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


def summarize_density_validation(rows: Iterable[Mapping]) -> dict:
    """Return descriptive density-level statistics without setting a threshold."""

    rows = list(rows)
    levels = sorted({int(row["n_injected_sources"]) for row in rows})
    summaries = []
    for level in levels:
        selected = [row for row in rows if int(row["n_injected_sources"]) == level]
        residuals = _finite_values(selected, "fractional_flux_residual")
        pulls = _finite_values(selected, "pull")
        changes = _finite_values(selected, "fractional_change_from_low_density")
        summaries.append(
            {
                "n_injected_sources": level,
                "injection_density_per_arcmin2": float(
                    selected[0]["injection_density_per_arcmin2"]
                ),
                "n_reference_sources": len(selected),
                "fractional_residual_median": _median_or_none(residuals),
                "fractional_residual_mad": _mad_or_none(residuals),
                "pull_median": _median_or_none(pulls),
                "pull_std": (
                    float(np.std(pulls, ddof=1)) if pulls.size > 1 else None
                ),
                "fractional_change_from_low_density_median": _median_or_none(changes),
                "fractional_change_from_low_density_max_abs": (
                    float(np.max(np.abs(changes))) if changes.size else None
                ),
                "measurement_flag_fraction": float(
                    np.mean([bool(row.get("measurement_failed")) for row in selected])
                ),
                "injection_failure_count": int(
                    max(row.get("injection_failure_count", 0) for row in selected)
                ),
                "level_runtime_seconds": float(selected[0]["level_runtime_seconds"]),
            }
        )
    return {"n_rows": len(rows), "levels": summaries}


def plot_density_validation(rows: Iterable[Mapping], output_dir: str | Path) -> None:
    """Create compact plots for the global injection-density question."""

    import matplotlib.pyplot as plt

    rows = list(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    density = np.asarray([r["injection_density_per_arcmin2"] for r in rows], dtype=float)
    residual = np.asarray([r["fractional_flux_residual"] for r in rows], dtype=float)
    pulls = np.asarray([r["pull"] for r in rows], dtype=float)
    changes = np.asarray(
        [r["fractional_change_from_low_density"] for r in rows], dtype=float
    )

    _scatter_by_magnitude(
        rows,
        density,
        residual,
        output_dir / "density_fractional_residual.png",
        "Injection density (sources / arcmin²)",
        "Fractional baseline-subtracted flux residual",
    )
    _scatter_by_magnitude(
        rows,
        density,
        pulls,
        output_dir / "density_pull.png",
        "Injection density (sources / arcmin²)",
        "Flux pull",
    )
    _scatter_by_magnitude(
        rows,
        density,
        changes,
        output_dir / "density_same_source_flux_change.png",
        "Injection density (sources / arcmin²)",
        "Same-source fractional flux change from lowest density",
    )

    summary = summarize_density_validation(rows)
    levels = summary["levels"]
    x = [level["injection_density_per_arcmin2"] for level in levels]
    flagged = [level["measurement_flag_fraction"] for level in levels]
    failed = [
        level["injection_failure_count"] / level["n_injected_sources"]
        for level in levels
    ]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(x, flagged, marker="o", label="Measurement flag fraction")
    axis.plot(x, failed, marker="s", label="Injection failure fraction")
    axis.set_xlabel("Injection density (sources / arcmin²)")
    axis.set_ylabel("Fraction")
    axis.set_ylim(bottom=0)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "density_failure_flag_fraction.png", dpi=140)
    plt.close(figure)


def _finite_values(rows: list[Mapping], key: str) -> np.ndarray:
    return np.asarray(
        [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(row[key])],
        dtype=float,
    )


def _median_or_none(values: np.ndarray) -> float | None:
    return float(np.median(values)) if values.size else None


def _mad_or_none(values: np.ndarray) -> float | None:
    if not values.size:
        return None
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def _scatter_by_magnitude(
    rows: list[Mapping],
    x: np.ndarray,
    y: np.ndarray,
    path: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 4))
    magnitudes = np.asarray([row["requested_mag"] for row in rows], dtype=float)
    for magnitude in sorted(set(magnitudes)):
        mask = magnitudes == magnitude
        axis.scatter(x[mask], y[mask], s=22, alpha=0.75, label=f"mag {magnitude:g}")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)

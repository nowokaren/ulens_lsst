import pytest

from ulens_lsst.density_validation import (
    DensityValidationDesign,
    add_low_density_comparisons,
)
from ulens_lsst.validation_analysis import summarize_density_validation


def test_density_design_requires_nested_increasing_counts():
    DensityValidationDesign(source_counts=(9, 50, 200)).validate()

    with pytest.raises(ValueError, match="increasing"):
        DensityValidationDesign(source_counts=(9, 200, 50)).validate()


def test_same_source_comparison_uses_lowest_density():
    rows = [
        {
            "image_id": "image",
            "source_id": "source",
            "n_injected_sources": 9,
            "recovered_injected_flux": 100.0,
        },
        {
            "image_id": "image",
            "source_id": "source",
            "n_injected_sources": 50,
            "recovered_injected_flux": 102.0,
        },
    ]

    result = add_low_density_comparisons(rows)

    assert result[0]["recovered_flux_change_from_low_density"] == 0.0
    assert result[1]["recovered_flux_change_from_low_density"] == 2.0
    assert result[1]["fractional_change_from_low_density"] == 0.02


def test_density_summary_reports_flags_failures_and_flux_change():
    rows = []
    for source, change, failed in (("a", 0.01, False), ("b", -0.02, True)):
        rows.append(
            {
                "source_id": source,
                "n_injected_sources": 50,
                "injection_density_per_arcmin2": 2.5,
                "fractional_flux_residual": change,
                "pull": change * 10,
                "fractional_change_from_low_density": change,
                "measurement_failed": failed,
                "injection_failure_count": 1,
                "level_runtime_seconds": 3.0,
            }
        )

    level = summarize_density_validation(rows)["levels"][0]

    assert level["n_reference_sources"] == 2
    assert level["measurement_flag_fraction"] == 0.5
    assert level["injection_failure_count"] == 1
    assert level["fractional_change_from_low_density_max_abs"] == 0.02

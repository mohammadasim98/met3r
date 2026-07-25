from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import met3r.upper_analysis as upper_analysis
from met3r.upper_analysis import (
    FIGURE_FILENAMES,
    TABLE_FILENAMES,
    _geometry_envelope,
    _naf_ablation_from_raw,
    _resolution_utility_from_raw,
    analyze_upper_run,
    prepare_primary_metric,
    robust_clean_normalize,
    summarize_attribution,
    summarize_controlled_fidelity,
    summarize_stress,
)


VARIANT = {
    "backbone": "dino",
    "feature_track": "user_37_grid_track",
    "feature_mode": "native",
    "resolution": 37,
}


def _controlled_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for shoe_index, shoe in enumerate(("a", "b", "c", "d")):
        for pair in ("left__back", "left__right"):
            first, second = pair.split("__")
            for changed_view in (first, second):
                clean = 0.1 + shoe_index * 0.01
                for corruption, severities in (
                    ("translation", (0.1, 0.2, 0.4)),
                    ("identity_swap", (1.0,)),
                    ("blur", (0.2, 0.5)),
                ):
                    for severity in severities:
                        effect = severity
                        rows.append(
                            {
                                **VARIANT,
                                "shoe_id": shoe,
                                "pair_id": pair,
                                "changed_view": changed_view,
                                "corruption": corruption,
                                "severity": severity,
                                "s00": clean,
                                "s10": clean + effect,
                                "s01": clean + 0.05,
                                "s11": clean + effect + 0.08,
                                "abstained": False,
                                "s10_first_to_second_overlap": 0.8,
                                "s10_second_to_first_overlap": 0.7,
                            }
                        )
    return pd.DataFrame(rows)


def _stress_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for shoe in ("a", "b", "c", "d"):
        rows.append(
            {
                **VARIANT,
                "shoe_id": shoe,
                "pair_id": "left__back",
                "s00": 0.1,
                "s10": 0.3,
                "s01": 0.15,
                "s11": 0.4,
                "abstained": False,
                "s10_first_to_second_overlap": 0.8,
                "s10_second_to_first_overlap": 0.7,
                "first_camera_preserved": True,
                "second_camera_preserved": True,
                "first_semantic_gate": True,
                "second_semantic_gate": True,
                "first_preservation_gate": True,
                "second_preservation_gate": True,
                "first_fit_relation": "improved",
                "second_fit_relation": "equivalent",
            }
        )
    return pd.DataFrame(rows)


def _calibration_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for shoe in ("a", "b", "c", "d"):
        for magnitude in (0.1, 0.2, 0.4):
            rows.append(
                {
                    **VARIANT,
                    "shoe_id": shoe,
                    "pair_id": "left__back",
                    "changed_view": "left",
                    "perturbation_kind": "rotation",
                    "magnitude": magnitude,
                    "geometry_effect": magnitude,
                    "overlap": 0.8,
                    "abstained": False,
                }
            )
    return pd.DataFrame(rows)


def test_primary_fidelity_uses_fixed_geometry_feature_effect() -> None:
    frame = _controlled_rows()
    prepared = prepare_primary_metric(frame)
    np.testing.assert_allclose(prepared["metric_delta"], frame["s10"] - frame["s00"])
    np.testing.assert_allclose(prepared["score"], frame["s10"])
    assert not np.array_equal(prepared["metric_delta"], frame["s11"] - frame["s00"])


def test_clean_normalization_is_within_backbone_variant() -> None:
    first = _controlled_rows()
    second = first.assign(backbone="other", s00=100.0, s10=101.0)
    normalized = robust_clean_normalize(pd.concat([first, second], ignore_index=True))
    groups = normalized.groupby(
        ["backbone", "feature_track", "feature_mode", "resolution"]
    )
    for _, group in groups:
        centers = group["clean_center"].unique()
        assert len(centers) == 1
        assert np.isfinite(group["clean_scale"]).all()
        assert (group["clean_scale"] >= 1e-6).all()


def test_identity_swap_has_auc_and_rank_but_no_severity_spearman() -> None:
    summary = summarize_controlled_fidelity(
        _controlled_rows().query("corruption == 'identity_swap'"),
        bootstrap_repeats=20,
    )
    row = summary.iloc[0]
    assert row["clean_corrupt_auc"] == pytest.approx(1.0)
    assert row["pairwise_rank_accuracy"] == pytest.approx(1.0)
    assert np.isnan(row["severity_spearman"])


def test_continuous_endpoints_average_placements_pairs_then_shoes() -> None:
    frame = _controlled_rows().query("corruption == 'translation'").copy()
    frame.loc[
        (frame["shoe_id"] == "a")
        & (frame["pair_id"] == "left__back")
        & (frame["changed_view"] == "left"),
        "s10",
    ] = [0.5, 0.3, 0.2]
    summary = summarize_controlled_fidelity(frame, bootstrap_repeats=0)
    assert summary.iloc[0]["n_shoes"] == 4
    assert summary.iloc[0]["n_directional_series"] == 16
    assert 0.8 < summary.iloc[0]["severity_spearman"] < 1.0


def test_abstention_and_small_samples_return_nan_instead_of_crashing() -> None:
    frame = _controlled_rows().query("corruption == 'translation'").head(1).copy()
    frame["abstained"] = True
    summary = summarize_controlled_fidelity(frame, bootstrap_repeats=0)
    row = summary.iloc[0]
    assert row["coverage"] == pytest.approx(0.0)
    assert np.isnan(row["severity_spearman"])
    assert np.isnan(row["pairwise_rank_accuracy"])


def test_bootstrap_requires_two_distinct_finite_shoes_per_endpoint() -> None:
    frame = _controlled_rows().query(
        "corruption == 'translation' and pair_id == 'left__back' "
        "and changed_view == 'left'"
    ).copy()
    frame["abstained"] = frame["shoe_id"] != "a"

    one_valid = summarize_controlled_fidelity(frame, bootstrap_repeats=100).iloc[0]

    assert one_valid["severity_spearman_n_shoes_valid"] == 1
    assert np.isnan(one_valid["severity_spearman_ci_low"])
    assert np.isnan(one_valid["severity_spearman_ci_high"])

    frame["abstained"] = ~frame["shoe_id"].isin(["a", "b"])
    two_valid = summarize_controlled_fidelity(frame, bootstrap_repeats=100).iloc[0]
    assert two_valid["severity_spearman_n_shoes_valid"] == 2
    assert np.isfinite(two_valid["severity_spearman_ci_low"])
    assert np.isfinite(two_valid["severity_spearman_ci_high"])


def test_attribution_ratios_weighted_absolute_effects_not_rowwise_ratios() -> None:
    frame = pd.DataFrame(
        [
            {
                **VARIANT,
                "shoe_id": shoe,
                "pair_id": "left__back",
                "changed_view": changed,
                "corruption": "translation",
                "s00": 0.0,
                "s10": feature,
                "s01": geometry,
                "s11": feature + geometry,
                "abstained": False,
            }
            for shoe in ("a", "b")
            for changed, feature, geometry in (
                ("left", 100.0, 0.0),
                ("back", 0.0, 1.0),
            )
        ]
    )

    row = summarize_attribution(frame, bootstrap_repeats=50).iloc[0]

    assert row["feature_effect_abs_mean"] == pytest.approx(50.0)
    assert row["geometry_effect_abs_mean"] == pytest.approx(0.5)
    assert row["absolute_effect_denominator"] == pytest.approx(50.5)
    assert row["absolute_feature_share"] == pytest.approx(100 / 101)
    assert row["absolute_feature_share_n_shoes_valid"] == 2


def test_geometry_envelope_interpolates_and_counts_missing_or_abstained_curves() -> None:
    observed = pd.DataFrame(
        [
            {
                **VARIANT,
                "shoe_id": "a",
                "pair_id": "left__back",
                "changed_view": "left",
                "corruption": "translation",
                "severity": 0.2,
                "s00": 0.0,
                "s10": 0.15,
                "s01": 0.0,
                "s11": 0.15,
                "abstained": False,
            }
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                **VARIANT,
                "shoe_id": "a",
                "pair_id": "left__back",
                "changed_view": "left",
                "perturbation_kind": kind,
                "magnitude": magnitude,
                "geometry_effect": effect,
                "abstained": abstained,
            }
            for kind, magnitude, effect, abstained in (
                ("rotation", 0.1, 0.1, False),
                ("rotation", 0.2, 0.2, False),
                ("relative_translation", 0.1, 0.2, True),
            )
        ]
    )

    summary = _geometry_envelope(
        observed,
        calibration,
        bootstrap_repeats=0,
        seed=42,
    )
    rotation = summary[summary["perturbation_kind"] == "rotation"].iloc[0]
    assert rotation["matched_magnitude_raw"] == pytest.approx(0.15)
    assert rotation["matched_magnitude_effective"] == pytest.approx(0.15)
    assert rotation["n_feature_cells_covered"] == 1
    for kind in ("relative_translation", "depth_scale", "relative_depth_noise"):
        row = summary[summary["perturbation_kind"] == kind].iloc[0]
        assert row["n_feature_cells_total"] == 1
        assert row["n_feature_cells_covered"] == 0
        assert row["n_feature_cells_censored"] == 1
        assert row["envelope_coverage"] == pytest.approx(0.0)
        assert np.isnan(row["matched_magnitude_raw"])


def test_stress_report_keeps_required_quality_and_fit_strata() -> None:
    outputs = summarize_stress(_stress_rows(), bootstrap_repeats=20)
    assert {
        "all",
        "both_camera_preserved",
        "both_semantic_gate",
        "both_preservation_gate",
        "mixed_fit_relation:improved|equivalent",
    } <= set(outputs["stratum"])
    fit = outputs[outputs["stratum"].str.contains("fit_relation")]
    assert set(fit["stratum"]) == {"mixed_fit_relation:improved|equivalent"}


def test_resolution_and_all_naf_contrasts_include_validity_endpoints() -> None:
    controlled = _controlled_rows().query(
        "corruption in ['blur', 'translation']"
    )
    variants = [controlled]
    for mode in (
        "bilinear",
        "naf_normal_guide",
        "naf_blurred_guide",
        "naf_shuffled_guide",
    ):
        variants.append(
            controlled.assign(
                feature_mode=mode,
                resolution=74,
                s10=controlled["s10"] + 0.01,
            )
        )
    frame = pd.concat(variants, ignore_index=True)

    resolution = _resolution_utility_from_raw(
        frame, bootstrap_repeats=20, seed=42
    )
    naf = _naf_ablation_from_raw(frame, bootstrap_repeats=20, seed=42)

    for endpoint in (
        "coverage",
        "abstention_rate",
        "first_to_second_overlap",
        "second_to_first_overlap",
        "nuisance_signed_delta",
    ):
        resolution_column = f"{endpoint}_difference_from_native"
        assert resolution_column in resolution
        assert f"{resolution_column}_n_shoes_valid" in resolution
        naf_column = f"{endpoint}_normal_minus_comparator"
        assert naf_column in naf
        assert f"{naf_column}_n_shoes_valid" in naf
    assert set(naf["comparator"]) == {
        "bilinear",
        "naf_blurred_guide",
        "naf_shuffled_guide",
    }


def test_severity_curve_weights_placements_pairs_and_shoes_equally() -> None:
    clean = pd.DataFrame(
        [
            {
                **VARIANT,
                "shoe_id": f"clean-{index}",
                "pair_id": "left__back",
                "changed_view": None,
                "corruption": "clean",
                "severity": 0.0,
                "s00": value,
                "s10": value,
                "s01": value,
                "s11": value,
                "abstained": False,
            }
            for index, value in enumerate((-1.0, 0.0, 1.0))
        ]
    )
    corrupt = pd.DataFrame(
        [
            {
                **VARIANT,
                "shoe_id": shoe,
                "pair_id": pair,
                "changed_view": "left",
                "corruption": "blur",
                "severity": 0.2,
                "s00": 0.0,
                "s10": effect,
                "s01": 0.0,
                "s11": effect,
                "abstained": False,
            }
            for shoe, pair, effect, repeats in (
                ("a", "left__back", 0.0, 10),
                ("a", "left__right", 2.0, 1),
                ("b", "left__back", 10.0, 1),
            )
            for _ in range(repeats)
        ]
    )

    curve = upper_analysis._severity_curve_rows(
        pd.concat([clean, corrupt], ignore_index=True)
    )
    row = curve.query("corruption == 'blur' and severity == 0.2").iloc[0]

    assert row["n_shoes"] == 2
    assert row["normalized_delta"] == pytest.approx(5.5 / 1.4826)


def test_semantic_plot_groups_never_connect_distinct_variants() -> None:
    frame = pd.DataFrame(
        [
            {
                **VARIANT,
                "corruption": "blur",
                "severity": severity,
                "normalized_delta": severity,
            }
            for severity in (0.0, 0.5)
        ]
        + [
            {
                **VARIANT,
                "feature_mode": "bilinear",
                "resolution": 74,
                "corruption": "blur",
                "severity": severity,
                "normalized_delta": severity + 1,
            }
            for severity in (0.0, 0.5)
        ]
    )

    groups = upper_analysis._semantic_plot_groups(
        frame,
        x="severity",
        group_columns=(*upper_analysis.VARIANT_COLUMNS, "corruption"),
    )

    assert len(groups) == 2
    for label, rows in groups:
        assert rows[list(VARIANT)].drop_duplicates().shape[0] == 1
        assert "feature_mode=" in label
        assert "resolution=" in label


def test_analyze_upper_run_writes_complete_named_snapshot(tmp_path: Path) -> None:
    generation = tmp_path / "metrics" / "aggregate_generations" / "generation"
    generation.mkdir(parents=True)
    controlled = _controlled_rows()
    upsampled = []
    for mode, offset in (
        ("bilinear", 0.01),
        ("naf_normal_guide", 0.04),
        ("naf_blurred_guide", 0.02),
        ("naf_shuffled_guide", -0.01),
    ):
        variant = controlled.assign(
            feature_mode=mode,
            resolution=74,
            s10=controlled["s10"] + offset,
        )
        upsampled.append(variant)
    pd.concat([controlled, *upsampled], ignore_index=True).to_parquet(
        generation / "controlled_fidelity.parquet"
    )
    _calibration_rows().to_parquet(generation / "geometry_calibration.parquet")
    _stress_rows().to_parquet(generation / "real_deformation_stress.parquet")
    (generation / "summary_preview.csv").write_text("analysis_track\n", encoding="utf-8")
    (tmp_path / "metrics" / "current").symlink_to(
        Path("aggregate_generations") / "generation",
        target_is_directory=True,
    )

    outputs = analyze_upper_run(tmp_path, bootstrap_repeats=20)

    assert {path.name for path in outputs.tables} == set(TABLE_FILENAMES)
    assert {path.name for path in outputs.figures} == set(FIGURE_FILENAMES)
    assert all(path.stat().st_size > 0 for path in (*outputs.tables, *outputs.figures))
    resolution = pd.read_csv(tmp_path / "reports/tables/resolution_utility.csv")
    assert "severity_spearman_difference_from_native_ci_low" in resolution
    naf = pd.read_csv(tmp_path / "reports/tables/naf_guide_ablation.csv")
    assert set(naf["comparator"]) == {
        "bilinear",
        "naf_blurred_guide",
        "naf_shuffled_guide",
    }
    assert "clean_corrupt_auc_normal_minus_comparator_ci_low" in naf
    expected_valid_counts = {
        "controlled_fidelity.csv": "severity_spearman_n_shoes_valid",
        "severity_curves.csv": "normalized_delta_n_shoes_valid",
        "backbone_comparison.csv": "severity_spearman_n_shoes_valid",
        "resolution_utility.csv": (
            "coverage_difference_from_native_n_shoes_valid"
        ),
        "naf_guide_ablation.csv": (
            "coverage_normal_minus_comparator_n_shoes_valid"
        ),
        "attribution.csv": "absolute_feature_share_n_shoes_valid",
        "geometry_calibration.csv": (
            "matched_magnitude_raw_n_shoes_valid"
        ),
        "real_deformation_stress.csv": "metric_delta_n_shoes_valid",
        "leave_one_shoe_out.csv": "severity_spearman_n_shoes_valid",
    }
    for filename, column in expected_valid_counts.items():
        assert column in pd.read_csv(tmp_path / "reports/tables" / filename)
    report = outputs.report_path.read_text(encoding="utf-8")
    assert report.startswith("# fixed/precomputed DA3 cameras (not calibrated GT)")
    assert "upper views only" in report
    assert "four shoes" in report
    assert "descriptive intervals" in report
    assert "Controlled fidelity track" in report
    assert "Geometry calibration track" in report
    assert "Real-deformation stress track" in report
    assert "confirmatory p-values" not in report.lower()

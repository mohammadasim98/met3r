from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import met3r.adjacent16_analysis as adjacent16_analysis
from met3r.job_registry import sha256_file
from met3r.adjacent16_analysis import (
    FIGURE_FILENAMES,
    REPORT_SECTION_TITLES,
    TABLE_STEMS,
    _shoe_clustered_interval,
    analyze_adjacent16_run,
    summarize_backbone_common37,
    summarize_analyzability,
    summarize_common_support,
    summarize_geometry_low_high,
    summarize_naf_vs_bilinear,
    summarize_resolution_bilinear,
    summarize_stress_reliability,
    summarize_support_coverage,
)


COMMON37_BACKBONES = (
    "dino_vitb16",
    "dinov2_vitb14",
    "dinov3_vitb16",
    "dinov3_vitl16_pixal3d",
)
SHOES = (
    "air_huarache_run_ultra_111449",
    "air_max_90_nrg_dc6083_500",
    "air_max_90_surplus_cq7743_300",
    "air_tuned_max_cv6984_001",
)
SHOE_A, SHOE_B, SHOE_C, SHOE_D = SHOES


def _common37_controlled_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    severities = {
        "translation": (0.03, 0.06, 0.12, 0.2),
        "thin_plate_warp": (0.25, 0.5, 0.75, 1.0),
        "jpeg": (0.25, 0.5, 0.75, 1.0),
        "identity_swap": (1.0,),
    }
    jpeg_effects = (-0.2, 0.1, -0.4, 0.3)
    for shoe in SHOES:
        for backbone in COMMON37_BACKBONES:
            variant = {
                "backbone": backbone,
                "feature_track": "user_37_grid_track",
                "feature_mode": "bilinear",
                "resolution": 37,
            }
            base = {
                **variant,
                "shoe_id": shoe,
                "quartet_id": "q0",
                "pair_id": f"{shoe}__q0__1__2",
                "changed_view": None,
                "corruption": "clean",
                "severity": 0.0,
                "baseline_steps": 1,
                "s00": 0.0,
                "s10": 0.0,
                "abstained": False,
            }
            rows.append(base)
            for corruption, strengths in severities.items():
                for index, severity in enumerate(strengths):
                    effect = (
                        jpeg_effects[index]
                        if corruption == "jpeg"
                        else float(severity)
                    )
                    abstained = (
                        shoe == SHOE_D
                        and backbone == "dino_vitb16"
                        and corruption == "translation"
                        and severity == 0.2
                    )
                    # The other backbones carry an extreme score at the exact
                    # key abstained by DINO. Exact common support must remove
                    # it from every backbone rather than average marginals.
                    if (
                        shoe == SHOE_D
                        and corruption == "translation"
                        and severity == 0.2
                        and backbone != "dino_vitb16"
                    ):
                        effect = 100.0
                    rows.append(
                        {
                            **variant,
                            "shoe_id": shoe,
                            "quartet_id": "q0",
                            "pair_id": f"{shoe}__q0__1__2",
                            "changed_view": "1",
                            "corruption": corruption,
                            "severity": severity,
                            "baseline_steps": 1,
                            "s00": 0.0,
                            "s10": np.nan if abstained else effect,
                            "abstained": abstained,
                        }
                    )
    return pd.DataFrame(rows)


def _dinov3l_controlled_rows(*, extra_shoe_a_pairs: int = 0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    severities = {
        "translation": (0.03, 0.06, 0.12, 0.2),
        "thin_plate_warp": (0.25, 0.5, 0.75, 1.0),
        "jpeg": (0.25, 0.5, 0.75, 1.0),
        "identity_swap": (1.0,),
    }
    resolution_offsets = {37: 0.0, 74: 0.01, 148: 0.02, 296: 0.03, 592: 0.04}
    variants = [
        ("bilinear", resolution)
        for resolution in (37, 74, 148, 296, 592)
    ] + [
        ("naf_normal_guide", resolution)
        for resolution in (74, 148, 296, 592)
    ]
    core_pairs = (("1", "2", 1), ("1", "3", 2), ("1", "4", 3))
    for shoe in SHOES:
        pairs = list(core_pairs)
        if shoe == SHOE_A:
            pairs.extend(
                ("1", str(10 + index), 1)
                for index in range(extra_shoe_a_pairs)
            )
        for first, second, baseline_steps in pairs:
            pair_id = f"{shoe}__q0__{first}__{second}"
            for feature_mode, resolution in variants:
                variant = {
                    "backbone": "dinov3_vitl16_pixal3d",
                    "feature_track": "user_37_grid_track",
                    "feature_mode": feature_mode,
                    "resolution": resolution,
                }
                rows.append(
                    {
                        **variant,
                        "shoe_id": shoe,
                        "quartet_id": "q0",
                        "pair_id": pair_id,
                        "changed_view": None,
                        "corruption": "clean",
                        "severity": 0.0,
                        "baseline_steps": baseline_steps,
                        "s00": 0.0,
                        "s10": 0.0,
                        "abstained": False,
                    }
                )
                for corruption, strengths in severities.items():
                    for index, severity in enumerate(strengths):
                        if corruption == "translation":
                            multiplier = {
                                SHOE_A: 1.0,
                                SHOE_B: -1.0,
                                SHOE_C: 0.0,
                                SHOE_D: 1.0,
                            }[shoe]
                            effect = multiplier * float(severity)
                        elif corruption == "thin_plate_warp":
                            effect = float(severity)
                        elif corruption == "identity_swap":
                            effect = 1.0
                        else:
                            effect = (
                                (0.1, 0.2, 0.3, 0.4)[index]
                                + resolution_offsets[resolution]
                                + (0.05 if feature_mode == "naf_normal_guide" else 0.0)
                            )
                        abstained = (
                            shoe == SHOE_D
                            and baseline_steps == 3
                            and feature_mode == "bilinear"
                            and resolution == 592
                            and corruption == "jpeg"
                            and severity == 1.0
                        )
                        if (
                            shoe == SHOE_D
                            and baseline_steps == 3
                            and resolution != 592
                            and corruption == "jpeg"
                            and severity == 1.0
                        ):
                            effect = (
                                100.05
                                if feature_mode == "naf_normal_guide"
                                else 100.0
                            )
                        rows.append(
                            {
                                **variant,
                                "shoe_id": shoe,
                                "quartet_id": "q0",
                                "pair_id": pair_id,
                                "changed_view": first,
                                "corruption": corruption,
                                "severity": severity,
                                "baseline_steps": baseline_steps,
                                "s00": 0.0,
                                "s10": np.nan if abstained else effect,
                                "abstained": abstained,
                            }
                        )
    return pd.DataFrame(rows)


def test_common37_endpoints_use_exact_row_intersection_and_four_shoe_units() -> None:
    """Catch an abstention becoming zero or unrelated backbone marginals being compared."""

    summary = summarize_backbone_common37(
        _common37_controlled_rows(),
        bootstrap_repeats=100,
        seed=42,
    )

    identity = summary.query(
        "baseline_steps == 'all' and corruption == 'identity_swap' "
        "and endpoint == 'clean_corrupt_auroc'"
    )
    assert set(identity["backbone"]) == set(COMMON37_BACKBONES)
    assert identity["estimate"].tolist() == pytest.approx([1.0] * 4)
    assert identity["n_shoes_valid"].tolist() == [4] * 4

    translation = summary.query(
        "baseline_steps == 'all' and corruption == 'translation'"
    )
    for endpoint in ("severity_spearman", "pairwise_rank_accuracy"):
        rows = translation.query("endpoint == @endpoint")
        assert rows["estimate"].tolist() == pytest.approx([1.0] * 4)
        assert rows["n_common_support"].tolist() == [15] * 4
        assert rows["n_union_support"].tolist() == [16] * 4
        assert rows["support_retention"].tolist() == pytest.approx([15 / 16] * 4)
        assert rows["complete_series_coverage"].tolist() == pytest.approx([3 / 4] * 4)

    for corruption in ("translation", "thin_plate_warp"):
        assert set(
            summary.query(
                "baseline_steps == 'all' and corruption == @corruption"
            )["endpoint"]
        ) == {"severity_spearman", "pairwise_rank_accuracy"}

    jpeg = summary.query(
        "baseline_steps == 'all' and corruption == 'jpeg'"
    )
    signed = jpeg.query("endpoint == 'nuisance_signed_delta'")
    absolute = jpeg.query("endpoint == 'nuisance_absolute_delta'")
    assert signed["estimate"].tolist() == pytest.approx([-0.05] * 4)
    assert absolute["estimate"].tolist() == pytest.approx([0.25] * 4)


def test_resolution_and_naf_differences_are_paired_on_exact_support_by_step() -> None:
    """Catch five-resolution or NAF contrasts formed from unrelated marginal means."""

    frame = _dinov3l_controlled_rows()
    resolution = summarize_resolution_bilinear(
        frame,
        bootstrap_repeats=100,
        seed=42,
    )
    naf = summarize_naf_vs_bilinear(
        frame,
        bootstrap_repeats=100,
        seed=42,
    )

    assert set(resolution["baseline_steps"]) == {"all", 1, 2, 3}
    signed = resolution.query(
        "baseline_steps == 'all' and corruption == 'jpeg' "
        "and endpoint == 'nuisance_signed_delta'"
    ).sort_values("resolution")
    assert signed["resolution"].tolist() == [37, 74, 148, 296, 592]
    assert signed["difference_from_37"].tolist() == pytest.approx(
        [0.0, 0.01, 0.02, 0.03, 0.04]
    )
    assert signed["difference_from_37_n_shoes_valid"].tolist() == [4] * 5
    assert signed["n_common_support"].tolist() == [47] * 5
    assert signed["n_union_support"].tolist() == [48] * 5
    assert signed["support_retention"].tolist() == pytest.approx([47 / 48] * 5)

    assert set(naf["resolution"]) == {74, 148, 296, 592}
    for steps in ("all", 1, 2, 3):
        paired = naf[
            (naf["baseline_steps"] == steps)
            & (naf["corruption"] == "jpeg")
            & (naf["endpoint"] == "nuisance_signed_delta")
        ]
        assert paired["naf_minus_bilinear"].tolist() == pytest.approx(
            [0.05] * 4
        )
        assert paired["naf_minus_bilinear_n_shoes_valid"].tolist() == [4] * 4


def test_bootstrap_resamples_shoes_not_repeated_pairs() -> None:
    """Catch quartet/pair rows being resampled as independent subjects."""

    balanced = summarize_resolution_bilinear(
        _dinov3l_controlled_rows(),
        bootstrap_repeats=500,
        seed=42,
    )
    repeated_a = summarize_resolution_bilinear(
        _dinov3l_controlled_rows(extra_shoe_a_pairs=4),
        bootstrap_repeats=500,
        seed=42,
    )
    columns = (
        "estimate",
        "ci_low",
        "ci_high",
        "n_shoes_valid",
    )
    query = (
        "baseline_steps == 'all' and resolution == 37 "
        "and corruption == 'translation' "
        "and endpoint == 'pairwise_rank_accuracy'"
    )
    first = balanced.query(query).iloc[0]
    second = repeated_a.query(query).iloc[0]

    assert first["estimate"] == pytest.approx(0.625)
    assert second[list(columns)].tolist() == pytest.approx(
        first[list(columns)].tolist()
    )


def test_bootstrap_keeps_an_entirely_missing_shoe_in_four_cluster_draws() -> None:
    """Catch an invalid shoe being dropped instead of retained as missing evidence."""

    shoes = pd.DataFrame(
        {
            "shoe_id": SHOES,
            "value": (0.0, 1.0, 2.0, np.nan),
        }
    )

    assert _shoe_clustered_interval(
        shoes,
        repeats=100,
        seed=42,
    ) == pytest.approx((0.0, 2.0))


def _geometry_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for shoe in SHOES:
        for steps in (1, 2, 3):
            for mode, resolution, multiplier in (
                ("bilinear", 37, 1.0),
                ("naf_normal_guide", 592, 2.0),
            ):
                for magnitude in (0.25, 0.5, 1.0, 2.0, 5.0):
                    abstained = (
                        shoe == SHOE_D
                        and steps == 3
                        and mode == "naf_normal_guide"
                        and magnitude == 2.0
                    )
                    effect = multiplier * magnitude
                    if (
                        shoe == SHOE_D
                        and steps == 3
                        and mode == "bilinear"
                        and magnitude == 2.0
                    ):
                        effect = 100.0
                    rows.append(
                        {
                            "backbone": "dinov3_vitl16_pixal3d",
                            "feature_track": "user_37_grid_track",
                            "feature_mode": mode,
                            "resolution": resolution,
                            "shoe_id": shoe,
                            "quartet_id": "q0",
                            "pair_id": f"{shoe}__q0__1__{steps + 1}",
                            "changed_view": "1",
                            "baseline_steps": steps,
                            "perturbation_kind": "rotation",
                            "magnitude": magnitude,
                            "base_score": 0.0,
                            "perturbed_score": np.nan if abstained else effect,
                            "geometry_effect": np.nan if abstained else effect,
                            "overlap": np.nan if abstained else 0.5,
                            "abstained": abstained,
                        }
                    )
    return pd.DataFrame(rows)


def _stress_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for shoe_index, shoe in enumerate(SHOES):
        for steps in (1, 2, 3):
            for mode, resolution in (
                ("bilinear", 37),
                ("naf_normal_guide", 592),
            ):
                high = mode == "naf_normal_guide"
                abstained = high and shoe == SHOE_D and steps == 3
                low_effect = 0.1 * (shoe_index + 1) + 0.01 * steps
                effect = low_effect + (0.2 if high else 0.0)
                rows.append(
                    {
                        "backbone": "dinov3_vitl16_pixal3d",
                        "feature_track": "user_37_grid_track",
                        "feature_mode": mode,
                        "resolution": resolution,
                        "shoe_id": shoe,
                        "quartet_id": "q0",
                        "pair_id": f"{shoe}__q0__1__{steps + 1}",
                        "baseline_steps": steps,
                        "s00": 0.0,
                        "s10": np.nan if abstained else effect,
                        "s01": 0.0,
                        "s11": np.nan if abstained else effect,
                        "score": np.nan if abstained else effect,
                        "feature_effect": np.nan if abstained else effect,
                        "s11_first_to_second_overlap": np.nan if abstained else 0.5,
                        "s11_second_to_first_overlap": np.nan if abstained else 0.4,
                        "abstained": abstained,
                        "first_alignment_class": (
                            "strict"
                            if shoe_index == 0
                            else "moderate"
                            if shoe_index in {1, 3}
                            else "high_misalignment"
                        ),
                        "second_alignment_class": (
                            "strict"
                            if shoe_index == 0
                            else "moderate"
                            if shoe_index in {1, 3}
                            else "high_misalignment"
                        ),
                        "first_center_shift_normalized": 0.005
                        + 0.01 * shoe_index,
                        "second_center_shift_normalized": 0.006
                        + 0.01 * shoe_index,
                        "first_bottom_shift_normalized": 0.004
                        + 0.01 * shoe_index,
                        "second_bottom_shift_normalized": 0.005
                        + 0.01 * shoe_index,
                        "first_width_ratio": 1.0 + 0.02 * shoe_index,
                        "second_width_ratio": 1.01 + 0.02 * shoe_index,
                        "first_same_shoe_identity": 0.99 - 0.02 * shoe_index,
                        "second_same_shoe_identity": 0.98 - 0.02 * shoe_index,
                    }
                )
    return pd.DataFrame(rows)


def _all_variant_controlled_rows() -> pd.DataFrame:
    dinov3l = _dinov3l_controlled_rows()
    common37_source = dinov3l[
        (dinov3l["feature_mode"] == "bilinear")
        & (dinov3l["resolution"] == 37)
    ]
    b_backbones = [
        common37_source.assign(backbone=backbone)
        for backbone in COMMON37_BACKBONES[:-1]
    ]
    return pd.concat([dinov3l, *b_backbones], ignore_index=True)


def _preflight_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for shoe in SHOES:
        for steps in (1, 2, 3):
            symmetric = not (shoe == SHOE_D and steps == 3)
            rows.append(
                {
                    "shoe_id": shoe,
                    "quartet_id": "q0",
                    "pair_id": f"{shoe}__q0__1__{steps + 1}",
                    "baseline_steps": steps,
                    "first_to_second_overlap": 0.5,
                    "second_to_first_overlap": 0.4,
                    "minimum_directional_overlap": 0.4,
                    "symmetric_pass": symmetric,
                }
            )
    return pd.DataFrame(rows)


def _write_validated_preflight(
    reports: Path,
    frame: pd.DataFrame,
) -> tuple[Path, Path]:
    reports.mkdir(parents=True, exist_ok=True)
    parquet = reports / "geometry_preflight.parquet"
    frame.to_parquet(parquet, index=False)
    decision = reports / "geometry_preflight.json"
    decision.write_text(
        json.dumps(
            {
                "passed": True,
                "output_sha256": sha256_file(parquet),
                "input_sha256": "synthetic-preflight-input",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return parquet, decision


def test_geometry_low_high_uses_exact_support_and_exposes_shoe_counts() -> None:
    """Catch an abstained high-resolution calibration cell becoming a low-score zero."""

    summary = summarize_geometry_low_high(
        _geometry_rows(),
        bootstrap_repeats=100,
        seed=42,
    )
    magnitude_two = summary.query(
        "baseline_steps == 'all' and perturbation_kind == 'rotation' "
        "and magnitude == 2.0"
    )

    assert set(magnitude_two["variant"]) == {"L37-bilinear", "L592-NAF"}
    assert magnitude_two.sort_values("variant")["geometry_response"].tolist() == pytest.approx(
        [2.0, 4.0]
    )
    assert magnitude_two["geometry_response_n_shoes_valid"].tolist() == [4, 4]
    assert magnitude_two["high_minus_low"].tolist() == pytest.approx([2.0, 2.0])
    assert magnitude_two["high_minus_low_n_shoes_valid"].tolist() == [4, 4]
    assert magnitude_two["n_common_support"].tolist() == [11, 11]
    assert magnitude_two["n_union_support"].tolist() == [12, 12]
    assert magnitude_two["support_retention"].tolist() == pytest.approx([11 / 12] * 2)
    step_three = summary.query(
        "baseline_steps == 3 and perturbation_kind == 'rotation' "
        "and magnitude == 2.0"
    )
    assert step_three["geometry_response_n_shoes_valid"].tolist() == [3, 3]
    assert step_three["high_minus_low_n_shoes_valid"].tolist() == [3, 3]
    magnitude_one = summary.query(
        "baseline_steps == 'all' and perturbation_kind == 'rotation' "
        "and magnitude == 1.0"
    )
    assert magnitude_one["support_retention"].tolist() == pytest.approx(
        [1.0] * 2
    )
    assert magnitude_one["complete_series_coverage"].tolist() == pytest.approx(
        [11 / 12] * 2
    )
    assert step_three["complete_series_coverage"].tolist() == pytest.approx(
        [3 / 4] * 2
    )


def test_stress_reliability_counts_low_finite_high_abstained_and_disagreement() -> None:
    """Catch stress abstention being imputed or finite disagreement using unpaired rows."""

    summary = summarize_stress_reliability(
        _stress_rows(),
        bootstrap_repeats=100,
        seed=42,
    )
    all_steps = summary[summary["baseline_steps"] == "all"].set_index("metric")

    assert all_steps.loc["low_symmetric_pass_rate", "estimate"] == pytest.approx(1.0)
    assert all_steps.loc["high_symmetric_pass_rate", "estimate"] == pytest.approx(11 / 12)
    assert all_steps.loc[
        "low_finite_high_abstained_rate", "estimate"
    ] == pytest.approx(1 / 12)
    assert all_steps.loc["both_finite_rate", "estimate"] == pytest.approx(11 / 12)
    high_pass = all_steps.loc["high_symmetric_pass_rate"]
    assert high_pass["strict_estimate"] == pytest.approx(1.0)
    assert high_pass["moderate_estimate"] == pytest.approx(5 / 6)
    assert high_pass["high_misalignment_estimate"] == pytest.approx(1.0)
    assert high_pass["strict_n_pairs"] == 3
    assert high_pass["moderate_n_pairs"] == 6
    assert high_pass["high_misalignment_n_pairs"] == 3
    assert {
        "spearman_disagreement_vs_center_shift",
        "spearman_disagreement_vs_bottom_shift",
        "spearman_disagreement_vs_width_deviation",
        "spearman_disagreement_vs_identity",
    } <= set(all_steps.index)
    assert all_steps.loc[
        "finite_low_high_disagreement", "estimate"
    ] == pytest.approx(0.2)
    assert all_steps.loc[
        "finite_low_high_disagreement", "n_shoes_valid"
    ] == 4
    assert all_steps["support_retention"].tolist() == pytest.approx(
        [1.0] * len(all_steps)
    )
    assert all_steps["complete_series_coverage"].tolist() == pytest.approx(
        [11 / 12] * len(all_steps)
    )
    assert set(summary["baseline_steps"]) == {"all", 1, 2, 3}


def test_module_design_interpretation_does_not_claim_an_unobserved_boundary() -> None:
    """Catch a zero low-finite/high-abstained rate being reported as direct evidence."""

    absent = pd.DataFrame(
        [
            {
                "baseline_steps": "all",
                "metric": "low_finite_high_abstained_rate",
                "estimate": 0.0,
            }
        ]
    )
    observed = absent.assign(estimate=1 / 12)

    absent_text = adjacent16_analysis._module_design_interpretation(absent)
    observed_text = adjacent16_analysis._module_design_interpretation(observed)

    assert "No low-finite/high-abstained cases were observed" in absent_text
    assert "does not provide direct evidence" in absent_text
    assert "observed rate was `0.0833`" in observed_text
    assert "candidate gating signal" in observed_text


def test_stress_analyzability_uses_only_exact_low_high_pairs() -> None:
    """Catch unrelated variants or unpaired low rows changing the stress criterion."""

    controlled = _all_variant_controlled_rows()
    stress = _stress_rows()
    unrelated = stress.iloc[[0]].copy()
    unrelated["backbone"] = "unrelated_backbone"
    unrelated["pair_id"] = f"{SHOE_A}__q0__1__99"
    unrelated["s11_first_to_second_overlap"] = 0.0
    unrelated["s11_second_to_first_overlap"] = 0.0
    unpaired_low = stress.iloc[[0]].copy()
    unpaired_low["pair_id"] = f"{SHOE_A}__q0__1__98"
    unpaired_low["s11_first_to_second_overlap"] = 0.0
    unpaired_low["s11_second_to_first_overlap"] = 0.0
    augmented = pd.concat(
        [stress, unrelated, unpaired_low],
        ignore_index=True,
    )
    common_support = pd.DataFrame(
        {
            "comparison": ["four_backbone_common37"],
            "n_shoes_valid": [4],
            "n_common_support": [1],
            "n_union_support": [1],
            "support_retention": [1.0],
            "complete_series_coverage": [1.0],
        }
    )

    summary = summarize_analyzability(
        controlled,
        _geometry_rows(),
        augmented,
        common_support,
    ).set_index("criterion")

    assert summary.loc[
        "stress_symmetric_pass_rate", "observed"
    ] == pytest.approx(23 / 24)
    assert summary.loc["stress_symmetric_pass_rate", "n_shoes_valid"] == 4


def test_completeness_is_stratified_by_baseline_not_copied_globally() -> None:
    """Catch global controlled completeness being copied into every step."""

    support = summarize_support_coverage(
        _preflight_rows(),
        _all_variant_controlled_rows(),
        _stress_rows(),
        bootstrap_repeats=20,
        seed=42,
    )
    symmetric = support[support["metric"] == "symmetric_coverage"].set_index(
        "baseline_steps"
    )

    assert symmetric.loc["all", "complete_series_coverage"] == pytest.approx(
        575 / 576
    )
    assert symmetric.loc[1, "complete_series_coverage"] == pytest.approx(1.0)
    assert symmetric.loc[2, "complete_series_coverage"] == pytest.approx(1.0)
    assert symmetric.loc[3, "complete_series_coverage"] == pytest.approx(
        191 / 192
    )


def test_all_summaries_emit_the_frozen_baseline_domain_when_step_is_absent() -> None:
    """Catch observed-only step enumeration hiding an entirely absent stratum."""

    controlled = _all_variant_controlled_rows()
    controlled = controlled[controlled["baseline_steps"] != 3]
    geometry = _geometry_rows()
    geometry = geometry[geometry["baseline_steps"] != 3]
    stress = _stress_rows()
    stress = stress[stress["baseline_steps"] != 3]
    preflight = _preflight_rows()
    preflight = preflight[preflight["baseline_steps"] != 3]

    backbone = summarize_backbone_common37(
        controlled,
        bootstrap_repeats=0,
        seed=42,
    )
    resolution = summarize_resolution_bilinear(
        controlled,
        bootstrap_repeats=0,
        seed=42,
    )
    naf = summarize_naf_vs_bilinear(
        controlled,
        bootstrap_repeats=0,
        seed=42,
    )
    geometry_summary = summarize_geometry_low_high(
        geometry,
        bootstrap_repeats=0,
        seed=42,
    )
    stress_summary = summarize_stress_reliability(
        stress,
        bootstrap_repeats=0,
        seed=42,
    )
    support = summarize_support_coverage(
        preflight,
        controlled,
        stress,
        bootstrap_repeats=0,
        seed=42,
    )
    common = summarize_common_support(
        backbone,
        resolution,
        naf,
        geometry_summary,
        stress_summary,
    )
    summaries = (
        backbone,
        resolution,
        naf,
        geometry_summary,
        stress_summary,
        support,
        common,
    )

    for summary in summaries:
        assert set(summary["baseline_steps"]) == {"all", 1, 2, 3}
        absent = summary[summary["baseline_steps"] == 3]
        assert (absent["n_common_support"] == 0).all()
        assert (absent["n_union_support"] == 0).all()
        assert (absent["support_retention"] == 0.0).all()
        assert (absent["complete_series_coverage"] == 0.0).all()


def test_inconsistent_baseline_steps_for_one_physical_key_are_rejected() -> None:
    """Catch duplicate key-to-step dictionary construction silently overwriting evidence."""

    controlled = _all_variant_controlled_rows()
    controlled_key = (
        (controlled["shoe_id"] == SHOE_A)
        & (controlled["pair_id"] == f"{SHOE_A}__q0__1__2")
        & (controlled["corruption"] == "translation")
        & (controlled["severity"] == 0.03)
        & (controlled["backbone"] == "dino_vitb16")
    )
    controlled.loc[controlled_key, "baseline_steps"] = 2
    with pytest.raises(ValueError, match="inconsistent baseline_steps"):
        summarize_backbone_common37(
            controlled,
            bootstrap_repeats=0,
            seed=42,
        )

    geometry = _geometry_rows()
    geometry_key = (
        (geometry["shoe_id"] == SHOE_A)
        & (geometry["baseline_steps"] == 1)
        & (geometry["magnitude"] == 0.25)
        & (geometry["feature_mode"] == "naf_normal_guide")
    )
    geometry.loc[geometry_key, "baseline_steps"] = 2
    with pytest.raises(ValueError, match="inconsistent baseline_steps"):
        summarize_geometry_low_high(
            geometry,
            bootstrap_repeats=0,
            seed=42,
        )

    stress = _stress_rows()
    stress_key = (
        (stress["shoe_id"] == SHOE_A)
        & (stress["baseline_steps"] == 1)
        & (stress["feature_mode"] == "naf_normal_guide")
    )
    stress.loc[stress_key, "baseline_steps"] = 2
    with pytest.raises(ValueError, match="inconsistent baseline_steps"):
        summarize_stress_reliability(
            stress,
            bootstrap_repeats=0,
            seed=42,
        )


def test_controlled_coverage_rejects_a_three_shoe_cohort() -> None:
    """Catch all-observed-shoes being mislabeled as the frozen all-four cohort."""

    controlled = _all_variant_controlled_rows()
    controlled = controlled[controlled["shoe_id"] != SHOE_D]

    with pytest.raises(ValueError, match="frozen four-shoe cohort"):
        summarize_support_coverage(
            _preflight_rows(),
            controlled,
            _stress_rows(),
            bootstrap_repeats=0,
            seed=42,
        )


def test_support_common_support_and_analyzability_are_artifact_derived() -> None:
    """Catch support criteria being copied from a gate report or losing four-shoe counts."""

    controlled = _all_variant_controlled_rows()
    geometry = _geometry_rows()
    stress = _stress_rows()
    backbone = summarize_backbone_common37(
        controlled,
        bootstrap_repeats=20,
        seed=42,
    )
    resolution = summarize_resolution_bilinear(
        controlled,
        bootstrap_repeats=20,
        seed=42,
    )
    naf = summarize_naf_vs_bilinear(
        controlled,
        bootstrap_repeats=20,
        seed=42,
    )
    geometry_summary = summarize_geometry_low_high(
        geometry,
        bootstrap_repeats=20,
        seed=42,
    )
    stress_summary = summarize_stress_reliability(
        stress,
        bootstrap_repeats=20,
        seed=42,
    )
    support = summarize_support_coverage(
        _preflight_rows(),
        controlled,
        stress,
        bootstrap_repeats=20,
        seed=42,
    )
    common = summarize_common_support(
        backbone,
        resolution,
        naf,
        geometry_summary,
        stress_summary,
    )
    analyzability = summarize_analyzability(
        controlled,
        geometry,
        stress,
        common,
    )

    symmetric = support.query(
        "baseline_steps == 'all' and metric == 'symmetric_coverage'"
    ).iloc[0]
    assert symmetric["estimate"] == pytest.approx(11 / 12)
    assert symmetric["n_shoes_valid"] == 4
    assert symmetric["complete_series_coverage"] == pytest.approx(575 / 576)
    assert set(support["baseline_steps"]) == {"all", 1, 2, 3}

    assert {
        "four_backbone_common37",
        "five_resolution_bilinear",
        "naf_vs_bilinear_74",
        "naf_vs_bilinear_148",
        "naf_vs_bilinear_296",
        "naf_vs_bilinear_592",
        "geometry_low_high",
        "stress_low_high",
    } == set(common["comparison"])
    assert common["n_shoes_valid"].between(0, 4).all()
    assert common["support_retention"].notna().all()
    assert common["complete_series_coverage"].notna().all()

    criteria = analyzability.set_index("criterion")
    assert criteria.loc[
        "controlled_complete_series_coverage", "observed"
    ] == pytest.approx(575 / 576)
    assert criteria.loc[
        "all_four_shoe_primary_cell_rate", "observed"
    ] == pytest.approx(143 / 144)
    controlled_comparisons = common[
        common["comparison"].isin(
            {
                "four_backbone_common37",
                "five_resolution_bilinear",
                "naf_vs_bilinear_74",
                "naf_vs_bilinear_148",
                "naf_vs_bilinear_296",
                "naf_vs_bilinear_592",
            }
        )
    ]
    assert common["support_retention"].min() < controlled_comparisons[
        "support_retention"
    ].min()
    assert criteria.loc[
        "minimum_common_support_retention", "observed"
    ] == pytest.approx(
        controlled_comparisons["support_retention"].min()
    )
    assert criteria.loc[
        "stress_symmetric_pass_rate", "observed"
    ] == pytest.approx(23 / 24)
    assert {"n_shoes_valid", "support_retention", "complete_series_coverage"} <= set(
        analyzability
    )


def test_analyze_adjacent16_run_publishes_exact_atomic_research_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch publication omitting a frozen table, figure group, report section, or coverage."""

    generation = (
        tmp_path
        / "metrics"
        / "aggregate_generations"
        / "successful-generation"
    )
    generation.mkdir(parents=True)
    controlled = _all_variant_controlled_rows()
    geometry = _geometry_rows()
    stress = _stress_rows()
    controlled.to_parquet(generation / "controlled_fidelity.parquet", index=False)
    geometry.to_parquet(generation / "geometry_calibration.parquet", index=False)
    stress.to_parquet(generation / "real_deformation_stress.parquet", index=False)
    (generation / "summary_preview.csv").write_text(
        "analysis_track\n",
        encoding="utf-8",
    )
    (tmp_path / "metrics" / "current").symlink_to(
        Path("aggregate_generations") / generation.name,
        target_is_directory=True,
    )
    reports = tmp_path / "reports"
    preflight_parquet, preflight_decision = _write_validated_preflight(
        reports,
        _preflight_rows(),
    )
    (reports / "stale-root.txt").write_text("stale\n", encoding="utf-8")
    (reports / "tables").mkdir()
    (reports / "tables/stale.csv").write_text("stale\n", encoding="utf-8")
    (reports / "figures").mkdir()
    (reports / "figures/stale.png").write_bytes(b"stale")

    outputs = analyze_adjacent16_run(
        tmp_path,
        bootstrap_repeats=20,
        seed=42,
    )

    assert {path.stem for path in outputs.tables} == set(TABLE_STEMS)
    assert {path.name for path in outputs.figures} == set(FIGURE_FILENAMES)
    assert outputs.report_path == tmp_path / "reports/final_research_report.md"
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.tables)
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.figures)
    assert (tmp_path / "reports/geometry_preflight.parquet").is_file()
    assert {
        path.name for path in (tmp_path / "reports").iterdir()
    } == {
        "geometry_preflight.parquet",
        "geometry_preflight.json",
        "tables",
        "figures",
        "final_research_report.md",
    }
    assert {
        path.name for path in (tmp_path / "reports/tables").iterdir()
    } == {
        f"{stem}.{suffix}"
        for stem in TABLE_STEMS
        for suffix in ("csv", "parquet")
    }
    assert {
        path.name for path in (tmp_path / "reports/figures").iterdir()
    } == set(FIGURE_FILENAMES)
    for stem in TABLE_STEMS:
        csv_path = tmp_path / "reports" / "tables" / f"{stem}.csv"
        parquet_path = tmp_path / "reports" / "tables" / f"{stem}.parquet"
        assert csv_path.is_file() and parquet_path.is_file()
        table = pd.read_parquet(parquet_path)
        assert any("n_shoes_valid" in column for column in table.columns)
        assert "support_retention" in table
        assert "complete_series_coverage" in table

    report = outputs.report_path.read_text(encoding="utf-8")
    assert report.startswith(
        "# 인접 Upper 16-view DINO Feature Fidelity 및 Reliability 연구 보고서"
    )
    assert [
        line.removeprefix("## ")
        for line in report.splitlines()
        if line.startswith("## ")
    ] == [
        f"{index}. {title}"
        for index, title in enumerate(REPORT_SECTION_TITLES, start=1)
    ]
    for filename in FIGURE_FILENAMES:
        assert f"](figures/{filename})" in report

    assert "한눈에 보는 결론" in report
    assert "관찰된 결과" in report
    assert "해석" in report
    assert "직접 확인된 사실" in report
    assert "이번 실험에서 확인되지 않은 것" in report
    assert "제안하는 모듈 설계" in report
    assert "0 ≤ pixel_weight ≤ semantic_weight ≤ 1" in report
    assert "fixed/precomputed DA3 camera (calibrated GT 아님)" in report
    assert "절대적인 DA3 카메라 오차를 추정하지 않는다" in report
    assert "해상도 hard gate가 직접 검증된 것은 아니다" in report
    assert "결측 또는 abstention을 0점으로 대체하지 않았다" in report
    assert f"controlled rows: {len(controlled):,}" in report
    assert f"geometry calibration rows: {len(geometry):,}" in report
    assert f"stress rows: {len(stress):,}" in report
    assert "seed: `42`" in report
    assert "bootstrap repeats: `20`" in report
    assert "sha256" in report.lower()
    assert sha256_file(preflight_parquet) in report

    published_paths = [
        preflight_parquet,
        preflight_decision,
        outputs.report_path,
        *outputs.tables,
        *outputs.figures,
        *[
            tmp_path / "reports" / "tables" / f"{stem}.parquet"
            for stem in TABLE_STEMS
        ],
    ]
    first_hashes = {
        path.relative_to(tmp_path): sha256_file(path)
        for path in published_paths
    }
    real_compute = adjacent16_analysis._compute_analysis_tables

    def mutate_preflight_after_snapshot(*args: object, **kwargs: object) -> object:
        tables = real_compute(*args, **kwargs)
        preflight_parquet.write_bytes(b"mutated after validated snapshot")
        return tables

    monkeypatch.setattr(
        adjacent16_analysis,
        "_compute_analysis_tables",
        mutate_preflight_after_snapshot,
    )
    second = analyze_adjacent16_run(
        tmp_path,
        bootstrap_repeats=20,
        seed=42,
    )
    monkeypatch.setattr(
        adjacent16_analysis,
        "_compute_analysis_tables",
        real_compute,
    )
    second_paths = [
        tmp_path / "reports" / "geometry_preflight.parquet",
        tmp_path / "reports" / "geometry_preflight.json",
        second.report_path,
        *second.tables,
        *second.figures,
        *[
            tmp_path / "reports" / "tables" / f"{stem}.parquet"
            for stem in TABLE_STEMS
        ],
    ]
    assert {
        path.relative_to(tmp_path): sha256_file(path)
        for path in second_paths
    } == first_hashes
    assert not list(tmp_path.glob(".reports.staging.*"))
    assert not list(tmp_path.glob(".reports.backup.*"))

    changed_preflight = _preflight_rows()
    changed_preflight.loc[0, "first_to_second_overlap"] = 0.25
    changed_parquet, _ = _write_validated_preflight(
        tmp_path / "reports",
        changed_preflight,
    )
    changed = analyze_adjacent16_run(
        tmp_path,
        bootstrap_repeats=20,
        seed=42,
    )
    assert sha256_file(changed.report_path) != first_hashes[
        outputs.report_path.relative_to(tmp_path)
    ]
    assert sha256_file(changed_parquet) in changed.report_path.read_text(
        encoding="utf-8"
    )


def test_analysis_rejects_preflight_whose_bytes_do_not_match_decision(
    tmp_path: Path,
) -> None:
    """Catch accepting mutable preflight bytes without validating their provenance hash."""

    generation = (
        tmp_path
        / "metrics"
        / "aggregate_generations"
        / "successful-generation"
    )
    generation.mkdir(parents=True)
    for filename in (
        "controlled_fidelity.parquet",
        "geometry_calibration.parquet",
        "real_deformation_stress.parquet",
    ):
        pd.DataFrame({"placeholder": [1]}).to_parquet(
            generation / filename,
            index=False,
        )
    (generation / "summary_preview.csv").write_text(
        "analysis_track\n",
        encoding="utf-8",
    )
    (tmp_path / "metrics" / "current").symlink_to(
        Path("aggregate_generations") / generation.name,
        target_is_directory=True,
    )
    parquet, _decision = _write_validated_preflight(
        tmp_path / "reports",
        _preflight_rows(),
    )
    parquet.write_bytes(b"unvalidated mutation")

    with pytest.raises(ValueError, match="not validated"):
        analyze_adjacent16_run(
            tmp_path,
            bootstrap_repeats=0,
            seed=42,
        )


def test_atomic_publish_rolls_back_when_install_rename_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a failed install leaving neither the old report nor retryable staging."""

    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "old.txt").write_text("old\n", encoding="utf-8")
    staging = tmp_path / ".reports.staging.test"
    staging.mkdir()
    (staging / "new.txt").write_text("new\n", encoding="utf-8")
    real_replace = adjacent16_analysis.os.replace
    calls = 0

    def fail_install(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected install rename failure")
        real_replace(source, destination)

    monkeypatch.setattr(adjacent16_analysis.os, "replace", fail_install)

    with pytest.raises(OSError, match="install rename"):
        adjacent16_analysis._publish_reports(tmp_path, staging)

    assert (reports / "old.txt").read_text(encoding="utf-8") == "old\n"
    assert (staging / "new.txt").read_text(encoding="utf-8") == "new\n"
    assert not list(tmp_path.glob(".reports.backup.*"))


def test_atomic_publish_restores_sole_backup_before_a_failed_install(
    tmp_path: Path,
) -> None:
    """Catch cleanup deleting the only recoverable snapshot when reports is absent."""

    backup = tmp_path / ".reports.backup.recoverable"
    backup.mkdir()
    (backup / "old.txt").write_text("old\n", encoding="utf-8")
    missing_staging = tmp_path / ".reports.staging.missing"

    with pytest.raises(OSError):
        adjacent16_analysis._publish_reports(tmp_path, missing_staging)

    assert (
        tmp_path / "reports" / "old.txt"
    ).read_text(encoding="utf-8") == "old\n"
    assert not missing_staging.exists()
    assert not list(tmp_path.glob(".reports.backup.*"))


def test_atomic_publish_fails_closed_on_ambiguous_backup_recovery(
    tmp_path: Path,
) -> None:
    """Catch multiple recovery candidates being deleted or selected arbitrarily."""

    first = tmp_path / ".reports.backup.first"
    second = tmp_path / ".reports.backup.second"
    first.mkdir()
    second.mkdir()
    (first / "old.txt").write_text("first\n", encoding="utf-8")
    (second / "old.txt").write_text("second\n", encoding="utf-8")
    staging = tmp_path / ".reports.staging.pending"
    staging.mkdir()
    (staging / "new.txt").write_text("new\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="ambiguous report recovery"):
        adjacent16_analysis._publish_reports(tmp_path, staging)

    assert not (tmp_path / "reports").exists()
    assert (first / "old.txt").read_text(encoding="utf-8") == "first\n"
    assert (second / "old.txt").read_text(encoding="utf-8") == "second\n"
    assert (staging / "new.txt").read_text(encoding="utf-8") == "new\n"


def test_atomic_publish_cleanup_failure_is_committed_and_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch post-install cleanup reporting failure after the new report committed."""

    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "old.txt").write_text("old\n", encoding="utf-8")
    staging = tmp_path / ".reports.staging.first"
    staging.mkdir()
    (staging / "new.txt").write_text("new\n", encoding="utf-8")
    real_rmtree = adjacent16_analysis.shutil.rmtree

    def fail_backup_cleanup(path: Path) -> None:
        if Path(path).name.startswith(".reports.backup."):
            raise OSError("injected backup cleanup failure")
        real_rmtree(path)

    monkeypatch.setattr(
        adjacent16_analysis.shutil,
        "rmtree",
        fail_backup_cleanup,
    )

    published = adjacent16_analysis._publish_reports(tmp_path, staging)

    assert (published / "new.txt").read_text(encoding="utf-8") == "new\n"
    assert list(tmp_path.glob(".reports.backup.*"))

    monkeypatch.setattr(adjacent16_analysis.shutil, "rmtree", real_rmtree)
    next_staging = tmp_path / ".reports.staging.second"
    next_staging.mkdir()
    (next_staging / "newer.txt").write_text("newer\n", encoding="utf-8")
    adjacent16_analysis._publish_reports(tmp_path, next_staging)
    assert (reports / "newer.txt").read_text(encoding="utf-8") == "newer\n"
    assert not list(tmp_path.glob(".reports.backup.*"))

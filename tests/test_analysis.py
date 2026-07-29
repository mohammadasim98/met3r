from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from met3r.analysis import (
    plot_fidelity_summary,
    summarize_attribution,
    summarize_fidelity,
)


def _results() -> pd.DataFrame:
    rows = []
    for shoe in ("a", "b"):
        for severity, score, human in zip(
            (0.0, 0.25, 0.5, 1.0),
            (0.05, 0.2, 0.4, 0.8),
            (5, 4, 3, 1),
        ):
            rows.append(
                {
                    "shoe_id": shoe,
                    "backbone": "dinov3",
                    "resolution": "37x37",
                    "corruption": "warp",
                    "severity": severity,
                    "score": score,
                    "human_consistency": human,
                    "abstained": False,
                    "overlap": 0.8,
                    "pair_scope": "intra_group",
                    "pair_id": "left__right",
                    "analysis_tier": "primary",
                    "semantic_identity": human,
                    "geometric_alignment": human,
                    "local_detail_alignment": human,
                    "single_view_plausibility": human,
                }
            )
    return pd.DataFrame(rows)


def test_fidelity_summary_reports_monotonicity_separation_and_human_agreement() -> None:
    summary = summarize_fidelity(_results())

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["severity_spearman"] == pytest.approx(1.0)
    assert row["clean_corrupt_auc"] == pytest.approx(1.0)
    assert row["pairwise_rank_accuracy"] == pytest.approx(1.0)
    assert row["human_semantic_identity_disagreement_spearman"] == pytest.approx(1.0)
    assert row["coverage"] == pytest.approx(1.0)
    assert row["clean_corrupt_auc_ci_low"] == pytest.approx(1.0)


def test_attribution_summary_keeps_geometry_and_feature_effects_separate() -> None:
    frame = _results().assign(
        feature_effect=0.3,
        geometry_effect=0.1,
        interaction=0.05,
        total_effect=0.45,
    )

    summary = summarize_attribution(frame, bootstrap_repeats=50)

    assert summary.iloc[0]["feature_effect_mean"] == pytest.approx(0.3)
    assert summary.iloc[0]["geometry_effect_mean"] == pytest.approx(0.1)
    assert summary.iloc[0]["feature_share_abs"] == pytest.approx(0.75)
    assert summary.iloc[0]["feature_effect_mean_ci_low"] == pytest.approx(0.3)


def test_attribution_equal_weights_pairs_then_shoes() -> None:
    frame = pd.DataFrame(
        [
            {
                "shoe_id": "a",
                "pair_id": "p1",
                "analysis_tier": "primary",
                "feature_effect": 1.0,
                "geometry_effect": 1.0,
                "interaction": 0.0,
                "total_effect": 2.0,
            },
            *[
                {
                    "shoe_id": "b",
                    "pair_id": "p1",
                    "analysis_tier": "primary",
                    "feature_effect": 9.0,
                    "geometry_effect": 1.0,
                    "interaction": 0.0,
                    "total_effect": 10.0,
                }
                for _ in range(9)
            ],
        ]
    )

    summary = summarize_attribution(frame, bootstrap_repeats=20)

    assert summary.iloc[0]["feature_effect_mean"] == pytest.approx(5.0)


def test_fidelity_plot_is_written(tmp_path: Path) -> None:
    output = tmp_path / "fidelity.png"
    plot_fidelity_summary(_results(), output)
    assert output.is_file()
    assert output.stat().st_size > 0

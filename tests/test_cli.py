from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from met3r import cli
from met3r.cli import _apply_mask_overrides, main


def test_camera_audit_uses_the_exact_camera_mode_label(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli,
        "build_bytesize_manifest",
        lambda _: SimpleNamespace(shoes=(), camera_provenance="fixed_precomputed_da3"),
    )

    summary = cli._camera_audit(tmp_path)

    assert summary["camera_mode"] == "fixed/precomputed DA3 cameras (not calibrated GT)"


def test_analyze_command_writes_machine_readable_summary_and_plot(tmp_path: Path) -> None:
    rows = []
    for shoe in ("a", "b"):
        for severity, score in ((0.0, 0.0), (0.5, 0.4), (1.0, 0.9)):
            rows.append(
                {
                    "shoe_id": shoe,
                    "backbone": "fake",
                    "resolution": "native",
                    "corruption": "warp",
                    "pair_scope": "intra_group",
                    "pair_id": "left__right",
                    "analysis_tier": "primary",
                    "severity": severity,
                    "score": score,
                    "overlap": 1.0,
                    "abstained": False,
                }
            )
    source = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    destination = tmp_path / "analysis"

    exit_code = main(["analyze", str(source), str(destination), "--bootstrap-repeats", "10"])

    assert exit_code == 0
    assert (destination / "fidelity_summary.csv").is_file()
    assert (destination / "score_vs_severity.png").is_file()


def test_mask_override_replaces_only_the_requested_view(tmp_path: Path) -> None:
    base = np.ones((2, 4, 5), dtype=bool)
    override = np.zeros((4, 5), dtype=np.uint8)
    override[1:3, 2:4] = 255
    override_path = tmp_path / "changed-mask.png"
    Image.fromarray(override).save(override_path)

    result = _apply_mask_overrides(base, [f"1={override_path}"])

    np.testing.assert_array_equal(result[0], base[0])
    np.testing.assert_array_equal(result[1], override > 127)


def test_mask_override_rejects_missing_camera_mask(tmp_path: Path) -> None:
    override_path = tmp_path / "changed-mask.png"
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(override_path)

    with pytest.raises(ValueError, match="base camera artifact"):
        _apply_mask_overrides(None, [f"0={override_path}"])

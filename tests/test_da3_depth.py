from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from met3r.da3_depth import DEFAULT_DA3_REVISION, load_da3_model, run_fixed_camera_depth


class _FakeDA3:
    def __init__(
        self,
        *,
        alter_pose: bool = False,
        alter_intrinsics: bool = False,
    ) -> None:
        self.kwargs: dict[str, object] | None = None
        self.alter_pose = alter_pose
        self.alter_intrinsics = alter_intrinsics

    def inference(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        extrinsics = np.asarray(kwargs["extrinsics"]).copy()
        if self.alter_pose:
            extrinsics[0, 0, 3] += 0.1
        count = extrinsics.shape[0]
        intrinsics = np.asarray(kwargs["intrinsics"]).copy()
        if self.alter_intrinsics:
            intrinsics[0, 0, 0] += 1.0
        return SimpleNamespace(
            depth=np.ones((count, 3, 4), dtype=np.float32),
            conf=np.full((count, 3, 4), 0.75, dtype=np.float32),
            processed_images=np.zeros((count, 3, 4, 3), dtype=np.uint8),
            extrinsics=extrinsics[:, :3],
            intrinsics=intrinsics,
        )


def _cameras(count: int = 2) -> tuple[np.ndarray, np.ndarray]:
    extrinsics = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    extrinsics[:, 0, 3] = np.arange(count)
    intrinsics = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    intrinsics[:, 0, 0] = 100.0
    intrinsics[:, 1, 1] = 100.0
    return extrinsics, intrinsics


def test_fixed_camera_depth_passes_poses_and_writes_provenance(tmp_path: Path) -> None:
    model = _FakeDA3()
    extrinsics, intrinsics = _cameras()
    foreground_mask = np.ones((2, 3, 4), dtype=bool)
    foreground_mask[:, 0] = False
    image_paths = []
    for name in ("a.png", "b.png"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        image_paths.append(path)

    artifact = run_fixed_camera_depth(
        model,
        image_paths,
        extrinsics,
        intrinsics,
        tmp_path,
        model_id="fake/da3",
        model_revision="fake-revision",
        source_state={
            "revision": "source-revision",
            "dirty": False,
            "dirty_fingerprint": None,
        },
        runtime_state={"python": "test", "packages": {}},
        camera_provenance="fixed_precomputed_da3",
        process_res=504,
        foreground_mask=foreground_mask,
        view_ids=("left", "right"),
        camera_groups=("upper", "upper"),
        analysis_tier="primary",
    )

    assert model.kwargs is not None
    np.testing.assert_allclose(model.kwargs["extrinsics"], extrinsics)
    np.testing.assert_allclose(model.kwargs["intrinsics"], intrinsics)
    assert model.kwargs["align_to_input_ext_scale"] is True
    arrays = np.load(artifact.arrays_path)
    np.testing.assert_allclose(arrays["depth"], 1.0)
    np.testing.assert_allclose(arrays["confidence"], 0.75)
    np.testing.assert_allclose(arrays["extrinsics"], extrinsics)
    np.testing.assert_array_equal(arrays["mask"], foreground_mask)
    assert arrays["meta__view_ids"].tolist() == ["left", "right"]
    assert arrays["meta__analysis_tier"].item() == "primary"
    metadata = json.loads(artifact.metadata_path.read_text())
    assert metadata["model_id"] == "fake/da3"
    assert metadata["model_revision"] == "fake-revision"
    assert metadata["source_state"]["revision"] == "source-revision"
    assert metadata["runtime_environment"]["python"] == "test"
    assert arrays["meta__runtime_fingerprint"].item()
    assert arrays["meta__source_revision"].item() == "source-revision"
    assert metadata["camera_provenance"] == "fixed_precomputed_da3"
    assert metadata["camera_mode"] == "fixed/precomputed DA3 cameras (not calibrated GT)"
    assert metadata["pose_max_abs_error"] == pytest.approx(0.0)
    assert metadata["mask_source"] == "provided_foreground_mask"


def test_depth_artifact_can_omit_processed_images(tmp_path: Path) -> None:
    model = _FakeDA3()
    extrinsics, intrinsics = _cameras()
    image_paths = []
    for name in ("a.png", "b.png"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        image_paths.append(path)

    artifact = run_fixed_camera_depth(
        model,
        image_paths,
        extrinsics,
        intrinsics,
        tmp_path,
        model_id="fake/da3",
        camera_provenance="fixed_precomputed_da3",
        store_processed_images=False,
    )

    with np.load(artifact.arrays_path) as arrays:
        assert "processed_images" not in arrays.files
        assert {"depth", "confidence", "mask", "extrinsics", "intrinsics"} <= set(
            arrays.files
        )
    metadata = json.loads(artifact.metadata_path.read_text())
    assert metadata["store_processed_images"] is False


def test_fixed_camera_depth_rejects_pose_drift(tmp_path: Path) -> None:
    extrinsics, intrinsics = _cameras()
    image_paths = []
    for name in ("a.png", "b.png"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        image_paths.append(path)

    with pytest.raises(ValueError, match="contract mismatch"):
        run_fixed_camera_depth(
            _FakeDA3(alter_pose=True),
            image_paths,
            extrinsics,
            intrinsics,
            tmp_path,
            model_id="fake/da3",
            camera_provenance="fixed_precomputed_da3",
        )


def test_fixed_camera_depth_rejects_intrinsics_drift(tmp_path: Path) -> None:
    extrinsics, intrinsics = _cameras()
    image_paths = []
    for name in ("a.png", "b.png"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        image_paths.append(path)

    with pytest.raises(ValueError, match="intrinsics"):
        run_fixed_camera_depth(
            _FakeDA3(alter_intrinsics=True),
            image_paths,
            extrinsics,
            intrinsics,
            tmp_path,
            model_id="fake/da3",
            camera_provenance="fixed_precomputed_da3",
        )


def test_da3_loader_passes_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None]] = []

    class FakeModel:
        @classmethod
        def from_pretrained(
            cls,
            model_id: str,
            *,
            revision: str | None,
        ) -> "FakeModel":
            calls.append((model_id, revision))
            return cls()

    fake_module = SimpleNamespace(DepthAnything3=FakeModel)
    monkeypatch.setitem(
        __import__("sys").modules,
        "depth_anything_3.api",
        fake_module,
    )

    model = load_da3_model("fake/da3")

    assert isinstance(model, FakeModel)
    assert calls == [("fake/da3", DEFAULT_DA3_REVISION)]

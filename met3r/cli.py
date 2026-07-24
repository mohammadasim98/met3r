"""Command-line entry points for the known-camera MEt3R experiment."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .analysis import plot_fidelity_summary, summarize_attribution, summarize_fidelity
from .bytesize_data import (
    build_bytesize_manifest,
    recover_bottom_camera_alignment,
    validate_processed_image,
    write_camera_artifacts,
)
from .corruptions import (
    CorruptionSpec,
    apply_corruption,
    apply_corruption_with_mask,
)
from .da3_depth import (
    DEFAULT_DA3_MODEL_ID,
    DEFAULT_DA3_REVISION,
    load_da3_model,
    run_fixed_camera_depth,
)
from .design import build_design_rows
from .human_rating import write_blinded_rating_packet
from .geometry_perturbations import write_geometry_calibration_artifacts
from .provenance import repository_state
from .runner import (
    extract_feature_artifact,
    score_feature_geometry_artifacts,
    score_geometry_sensitivity_artifacts,
)


def _integer_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(item) for item in value.split(",")]


def _float_list(value: str) -> list[float]:
    if not value:
        return []
    return [float(item) for item in value.split(",")]


def _read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_image_spec(path: str | Path) -> tuple[list[object], dict[str, object]]:
    source = Path(path)
    payload = _read_json(source)
    if isinstance(payload, list):
        images = payload
        context: dict[str, object] = {}
    elif isinstance(payload, dict) and isinstance(payload.get("images"), list):
        images = payload["images"]
        context = {key: value for key, value in payload.items() if key != "images"}
    else:
        raise ValueError("images_json must contain a list or an object with an images list")
    sibling_metadata = source.parent / "metadata.json"
    if sibling_metadata.is_file():
        metadata = _read_json(sibling_metadata)
        if isinstance(metadata, dict):
            context = {**metadata, **context}
    if "views" in context and isinstance(context["views"], list):
        context.setdefault(
            "view_ids",
            [item["view_id"] for item in context["views"] if isinstance(item, dict)],
        )
        context.setdefault(
            "camera_groups",
            [item["camera_group"] for item in context["views"] if isinstance(item, dict)],
        )
    return images, context


def _apply_mask_overrides(
    foreground_mask: np.ndarray | None,
    overrides: Sequence[str],
) -> np.ndarray | None:
    """Replace selected view masks from `INDEX=IMAGE` specifications."""

    if not overrides:
        return foreground_mask
    if foreground_mask is None:
        raise ValueError(
            "--mask-override requires a mask in the base camera artifact"
        )
    result = np.asarray(foreground_mask, dtype=bool).copy()
    if result.ndim != 3:
        raise ValueError("base camera mask must have shape [views, height, width]")
    height, width = result.shape[-2:]
    seen: set[int] = set()
    for specification in overrides:
        index_text, separator, path_text = specification.partition("=")
        if not separator:
            raise ValueError("--mask-override must use INDEX=PATH")
        try:
            index = int(index_text)
        except ValueError as error:
            raise ValueError("--mask-override index must be an integer") from error
        if not 0 <= index < result.shape[0]:
            raise IndexError(f"mask override index {index} is outside view count")
        if index in seen:
            raise ValueError(f"duplicate mask override for view {index}")
        seen.add(index)
        with Image.open(Path(path_text).expanduser()) as image:
            replacement = np.asarray(
                image.convert("L").resize(
                    (width, height),
                    Image.Resampling.NEAREST,
                )
            ) > 127
        result[index] = replacement
    return result


def _camera_audit(reconstruction_root: str | Path) -> dict[str, object]:
    manifest = build_bytesize_manifest(reconstruction_root)
    shoes = []
    for shoe in manifest.shoes:
        correlations = {}
        processed_cache: dict[Path, np.ndarray] = {}
        for view in shoe.views:
            processed_cache.setdefault(
                view.processed_images_path,
                np.load(view.processed_images_path)["processed_images"],
            )
            correlations[view.view_id] = validate_processed_image(
                view.image_path,
                processed_cache[view.processed_images_path],
                camera_index=view.camera_index,
            )
        upper = next(view for view in shoe.views if view.camera_group == "upper")
        bottom = next(view for view in shoe.views if view.camera_group == "bottom")
        upper_raw = np.load(upper.raw_extrinsics_path)["extrinsics"]
        bottom_raw = np.load(bottom.raw_extrinsics_path)["extrinsics"]
        bottom_custom = np.load(bottom.custom_extrinsics_path)["extrinsics"]
        correction = recover_bottom_camera_alignment(
            upper_raw,
            bottom_raw,
            bottom_custom,
        )
        pairs = []
        for first, second in itertools.combinations(shoe.views, 2):
            same_group = first.camera_group == second.camera_group
            pairs.append(
                {
                    "first_view": first.view_id,
                    "second_view": second.view_id,
                    "scope": "intra_group" if same_group else "cross_group",
                    "analysis_tier": "primary" if same_group else "sensitivity_only",
                    "camera_provenance": (
                        correction.intra_group_provenance
                        if same_group
                        else correction.cross_group_provenance
                    ),
                }
            )
        shoes.append(
            {
                "shoe_id": shoe.shoe_id,
                "view_image_correlations": correlations,
                "minimum_image_correlation": min(correlations.values()),
                "transform_residual": correction.transform_residual,
                "bottom_to_upper_rotation": correction.bottom_to_upper_rotation.tolist(),
                "exported_notebook_transform": correction.exported_transform.tolist(),
                "cross_group_translation_is_proxy": correction.translation_is_proxy,
                "pairs": pairs,
            }
        )
    return {
        "camera_mode": "fixed/precomputed (not calibrated GT)",
        "camera_provenance": manifest.camera_provenance,
        "primary_inference_scope": "intra_group pairs",
        "cross_group_policy": (
            "sensitivity analysis only: GLB centering translation was not saved, so the "
            "recovered rotation uses a translation proxy"
        ),
        "shoes": shoes,
    }


def _add_commands(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("prepare-manifest")
    manifest.add_argument("reconstruction_root")
    manifest.add_argument("output")
    manifest.add_argument("--deformed-root")

    audit = commands.add_parser("audit-cameras")
    audit.add_argument("reconstruction_root")
    audit.add_argument("output")

    camera_artifacts = commands.add_parser("prepare-camera-artifacts")
    camera_artifacts.add_argument("reconstruction_root")
    camera_artifacts.add_argument("output_dir")

    design = commands.add_parser("build-design")
    design.add_argument("config_json")
    design.add_argument("manifest_json")
    design.add_argument("output_csv")

    corrupt = commands.add_parser("corrupt")
    corrupt.add_argument("input")
    corrupt.add_argument("output")
    corrupt.add_argument("--kind", required=True)
    corrupt.add_argument("--strength", type=float, required=True)
    corrupt.add_argument("--seed", type=int, default=42)
    corrupt.add_argument("--donor")
    corrupt.add_argument("--part-mask")
    corrupt.add_argument("--foreground-mask")
    corrupt.add_argument("--output-mask")

    ratings = commands.add_parser("make-ratings")
    ratings.add_argument("samples_json")
    ratings.add_argument("output_dir")
    ratings.add_argument("--seed", type=int, default=42)
    ratings.add_argument("--repeat-fraction", type=float, default=0.1)

    analyze = commands.add_parser("analyze")
    analyze.add_argument("results_csv")
    analyze.add_argument("output_dir")
    analyze.add_argument("--bootstrap-repeats", type=int, default=500)
    analyze.add_argument("--seed", type=int, default=42)

    features = commands.add_parser("extract-features")
    features.add_argument("images_json")
    features.add_argument("output_dir")
    features.add_argument("--backbone", required=True)
    features.add_argument("--input-size", type=int, default=592)
    features.add_argument("--bilinear-scales", type=_integer_list, default=[])
    features.add_argument("--naf-scales", type=_integer_list, default=[])
    features.add_argument("--normal-naf-guide-only", action="store_true")
    features.add_argument("--device", default="cuda")

    score = commands.add_parser("score-artifacts")
    score.add_argument("clean_features")
    score.add_argument("changed_features")
    score.add_argument("base_geometry")
    score.add_argument("adaptive_geometry")
    score.add_argument("output_csv")
    score.add_argument("--first-index", type=int, required=True)
    score.add_argument("--second-index", type=int, required=True)
    score.add_argument("--metadata-json")
    score.add_argument("--device", default="cpu")
    score.add_argument("--confidence-percentile", type=float, default=10.0)

    sensitivity = commands.add_parser("score-geometry-sensitivity")
    sensitivity.add_argument("features")
    sensitivity.add_argument("base_geometry")
    sensitivity.add_argument("perturbed_geometry")
    sensitivity.add_argument("output_csv")
    sensitivity.add_argument("--first-index", type=int, required=True)
    sensitivity.add_argument("--second-index", type=int, required=True)
    sensitivity.add_argument("--metadata-json")
    sensitivity.add_argument("--device", default="cpu")
    sensitivity.add_argument("--confidence-percentile", type=float, default=10.0)

    depth = commands.add_parser("infer-depth")
    depth.add_argument("images_json")
    depth.add_argument("cameras_npz")
    depth.add_argument("output_dir")
    depth.add_argument("--model-id", default=DEFAULT_DA3_MODEL_ID)
    depth.add_argument("--model-revision", default=DEFAULT_DA3_REVISION)
    depth.add_argument("--da3-repository")
    depth.add_argument("--camera-provenance", required=True)
    depth.add_argument("--process-res", type=int, default=504)
    depth.add_argument(
        "--mask-override",
        action="append",
        default=[],
        metavar="INDEX=PATH",
        help="replace one camera mask, e.g. with a spatially corrupted output mask",
    )
    depth.add_argument("--device", default="cuda")

    calibration = commands.add_parser("calibrate-geometry")
    calibration.add_argument("base_geometry")
    calibration.add_argument("output_dir")
    calibration.add_argument("--view-index", type=int, required=True)
    calibration.add_argument("--rotations", type=_float_list, default=[])
    calibration.add_argument("--relative-translations", type=_float_list, default=[])
    calibration.add_argument("--depth-scales", type=_float_list, default=[])
    calibration.add_argument("--relative-depth-noise", type=_float_list, default=[])
    calibration.add_argument("--seed", type=int, default=42)


def _run(args: argparse.Namespace) -> None:
    if args.command == "prepare-manifest":
        result = build_bytesize_manifest(
            args.reconstruction_root,
            deformed_root=args.deformed_root,
        )
        result.write_json(Path(args.output))
        return
    if args.command == "audit-cameras":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(_camera_audit(args.reconstruction_root), indent=2),
            encoding="utf-8",
        )
        return
    if args.command == "prepare-camera-artifacts":
        manifest = build_bytesize_manifest(args.reconstruction_root)
        write_camera_artifacts(manifest, args.output_dir)
        return
    if args.command == "build-design":
        config = _read_json(args.config_json)
        manifest = _read_json(args.manifest_json)
        if not isinstance(config, dict) or not isinstance(manifest, dict):
            raise ValueError("config_json and manifest_json must contain objects")
        rows = build_design_rows(config, manifest)
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
        return
    if args.command == "corrupt":
        with Image.open(args.input) as image:
            source = np.asarray(image.convert("RGB"))
        donor = None
        mask = None
        if args.donor:
            with Image.open(args.donor) as image:
                donor = np.asarray(image.convert("RGB").resize((source.shape[1], source.shape[0])))
        if args.part_mask:
            with Image.open(args.part_mask) as image:
                mask = np.asarray(
                    image.convert("L").resize(
                        (source.shape[1], source.shape[0]), Image.Resampling.NEAREST
                    )
                ) > 127
        if bool(args.foreground_mask) != bool(args.output_mask):
            raise ValueError("--foreground-mask and --output-mask must be provided together")
        if args.foreground_mask:
            with Image.open(args.foreground_mask) as image:
                foreground_mask = np.asarray(
                    image.convert("L").resize(
                        (source.shape[1], source.shape[0]),
                        Image.Resampling.NEAREST,
                    )
                ) > 127
            result, changed_mask = apply_corruption_with_mask(
                source,
                foreground_mask,
                CorruptionSpec(args.kind, args.strength),
                seed=args.seed,
                donor=donor,
                part_mask=mask,
            )
            mask_output = Path(args.output_mask)
            mask_output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(changed_mask.astype(np.uint8) * 255).save(mask_output)
        else:
            result = apply_corruption(
                source,
                CorruptionSpec(args.kind, args.strength),
                seed=args.seed,
                donor=donor,
                part_mask=mask,
            )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result).save(output)
        return
    if args.command == "make-ratings":
        samples = _read_json(args.samples_json)
        if not isinstance(samples, list):
            raise ValueError("samples_json must contain a list")
        write_blinded_rating_packet(
            samples,
            args.output_dir,
            seed=args.seed,
            repeat_fraction=args.repeat_fraction,
        )
        return
    if args.command == "analyze":
        frame = pd.read_csv(args.results_csv)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        summarize_fidelity(
            frame,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed,
        ).to_csv(output / "fidelity_summary.csv", index=False)
        attribution_columns = {
            "feature_effect",
            "geometry_effect",
            "interaction",
            "total_effect",
        }
        if attribution_columns <= set(frame.columns):
            summarize_attribution(
                frame,
                bootstrap_repeats=args.bootstrap_repeats,
                seed=args.seed,
            ).to_csv(
                output / "attribution_summary.csv", index=False
            )
        plot_frame = (
            frame[~frame["abstained"].astype(bool)]
            if "abstained" in frame
            else frame
        )
        plot_fidelity_summary(
            plot_frame,
            output / "score_vs_severity.png",
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed,
        )
        return
    if args.command == "extract-features":
        images, context = _read_image_spec(args.images_json)
        extract_feature_artifact(
            images,
            args.output_dir,
            backbone_name=args.backbone,
            input_size=args.input_size,
            bilinear_scales=args.bilinear_scales,
            naf_scales=args.naf_scales,
            naf_guide_ablations=not args.normal_naf_guide_only,
            device=args.device,
            view_ids=context.get("view_ids"),
            camera_groups=context.get("camera_groups"),
            camera_provenance=context.get("camera_provenance"),
            analysis_tier=context.get("analysis_tier"),
        )
        return
    if args.command == "score-artifacts":
        metadata = _read_json(args.metadata_json) if args.metadata_json else {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must contain an object")
        rows = score_feature_geometry_artifacts(
            args.clean_features,
            args.changed_features,
            args.base_geometry,
            args.adaptive_geometry,
            first_index=args.first_index,
            second_index=args.second_index,
            metadata=metadata,
            device=args.device,
            confidence_percentile=args.confidence_percentile,
            strict_metadata=True,
        )
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
        return
    if args.command == "score-geometry-sensitivity":
        metadata = _read_json(args.metadata_json) if args.metadata_json else {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must contain an object")
        rows = score_geometry_sensitivity_artifacts(
            args.features,
            args.base_geometry,
            args.perturbed_geometry,
            first_index=args.first_index,
            second_index=args.second_index,
            metadata=metadata,
            device=args.device,
            confidence_percentile=args.confidence_percentile,
            strict_metadata=True,
        )
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
        return
    if args.command == "infer-depth":
        images, image_context = _read_image_spec(args.images_json)
        with np.load(args.cameras_npz) as cameras:
            extrinsics = np.asarray(cameras["extrinsics"])
            intrinsics = np.asarray(cameras["intrinsics"])
            foreground_mask = (
                np.asarray(cameras["mask"], dtype=bool)
                if "mask" in cameras
                else None
            )
            view_ids = (
                np.asarray(cameras["view_ids"]).tolist()
                if "view_ids" in cameras
                else image_context.get("view_ids")
            )
            camera_groups = (
                np.asarray(cameras["camera_groups"]).tolist()
                if "camera_groups" in cameras
                else image_context.get("camera_groups")
            )
        foreground_mask = _apply_mask_overrides(
            foreground_mask,
            args.mask_override,
        )
        camera_metadata_path = Path(args.cameras_npz).parent / "metadata.json"
        camera_metadata = (
            _read_json(camera_metadata_path)
            if camera_metadata_path.is_file()
            else {}
        )
        if not isinstance(camera_metadata, dict):
            camera_metadata = {}
        analysis_tier = camera_metadata.get(
            "analysis_tier", image_context.get("analysis_tier")
        )
        model = load_da3_model(
            args.model_id,
            revision=args.model_revision,
            repository_root=args.da3_repository,
            device=args.device,
        )
        source_state = (
            repository_state(args.da3_repository)
            if args.da3_repository is not None
            else None
        )
        run_fixed_camera_depth(
            model,
            images,
            extrinsics,
            intrinsics,
            args.output_dir,
            model_id=args.model_id,
            model_revision=args.model_revision,
            camera_provenance=args.camera_provenance,
            process_res=args.process_res,
            foreground_mask=foreground_mask,
            view_ids=view_ids,
            camera_groups=camera_groups,
            analysis_tier=analysis_tier,
            source_state=source_state,
        )
        return
    if args.command == "calibrate-geometry":
        write_geometry_calibration_artifacts(
            args.base_geometry,
            args.output_dir,
            view_index=args.view_index,
            rotation_degrees=args.rotations,
            translation_fractions=args.relative_translations,
            depth_scales=args.depth_scales,
            depth_noise_stds=args.relative_depth_noise,
            seed=args.seed,
        )
        return
    raise AssertionError(f"unknown command {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="met3r-known-camera")
    _add_commands(parser)
    args = parser.parse_args(argv)
    _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

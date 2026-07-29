# MEt3R Full-Run GPU Runtime Override Design

## Goal

Run the approved `20260724_upper_v2` full experiment on physical GPUs
`0,1,2,3` because GPUs `4,5,6,7` are occupied, while preserving the passing
GPU-1 pilot, its scientific configuration, and the tag
`upper-dino-fidelity-v2-pilot-passed`.

## Constraints

- Do not change image states, views, pairs, perturbations, feature variants,
  metric thresholds, model revisions, seeds, or camera artifacts.
- Continue to report cameras as
  `fixed/precomputed DA3 cameras (not calibrated GT)`.
- Do not alter or rerun the passing pilot.
- Record the actual full-run GPU assignments in `run_manifest.json`.
- Keep geometry and feature phases sequential; allow four workers in parallel
  only within a phase.
- Do not stage, edit, or remove `configs/environment_pixal3d.json`.

## Interface

Extend the prepare command with:

```text
met3r.upper_cli prepare CONFIG --scope full --physical-gpus 0,1,2,3
```

`--physical-gpus` is optional. Without it, preparation retains the frozen
configuration's `physical_gpus` value. With it, preparation uses the supplied
four physical device indices only for the operational run policy written to
`run_manifest.json`.

The full launcher exposes:

```bash
FULL_GPUS=${FULL_GPUS:-0,1,2,3}
```

It passes the value to `prepare --physical-gpus` and maps the four parsed
indices, in order, to the four shoes during geometry and the four backbones
during feature extraction.

## Validation

The override must contain exactly four unique, non-negative integer device
indices. Invalid input fails before workers start. The run manifest records:

- `physical_gpus`: `[0, 1, 2, 3]`;
- shoe-to-GPU geometry assignments in frozen shoe order;
- backbone-to-GPU feature assignments in frozen backbone order;
- `source`: `runtime_override`;
- unchanged `phase_order`: `["geometry", "feature"]`;
- unchanged `allow_phase_overlap`: `false`;
- unchanged pilot provenance `pilot_gpu: 1`.

When no override is supplied, `source` is `frozen_config` and the current
configuration behavior remains compatible.

An existing run tree remains immutable. Re-preparing it with a different
operational GPU policy must fail rather than silently retain stale provenance.

## Data Flow

1. The CLI parses the comma-separated GPU override into four integers.
2. `prepare_run` validates the operational GPU tuple separately from the
   frozen scientific configuration.
3. `_write_run_manifest` binds the validated tuple and its source to the run
   manifest.
4. The launcher parses the same `FULL_GPUS` value and uses the corresponding
   physical index in `CUDA_VISIBLE_DEVICES`.
5. Workers continue to receive `--device cuda:0`, so each process sees only
   its assigned physical device.

## Testing

Use test-driven development:

1. A CLI/prepare test must fail until the override reaches the run manifest.
2. Validation tests cover wrong cardinality, duplicate indices, negative
   indices, and non-integer input.
3. An immutability test verifies that a conflicting override cannot reuse an
   existing run root.
4. A behavioral launcher test executes a safe copy with a fake Python command
   and verifies the exact `0,1,2,3` mapping for both phases.
5. Run focused tests, shell syntax checks, the full test suite, and
   `git diff --check` before starting the real full experiment.

## Non-Goals

- Dynamic GPU scheduling or auto-selection.
- Pausing or terminating other GPU workloads.
- Changing the pilot tag or scientific configuration hash.
- Running geometry and feature phases concurrently.

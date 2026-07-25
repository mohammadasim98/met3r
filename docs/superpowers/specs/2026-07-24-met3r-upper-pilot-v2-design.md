# MEt3R Upper Pilot V2 Protocol Revision

## Status

Approved direction: option 1, an overlap-aware pilot with a geometrically
eligible primary pair and an expected-abstention negative-control pair.

This revision is frozen before inspecting any finite DINO metric score. It is
based only on the V1 run-completeness checks and clean fixed/precomputed DA3
geometry diagnostics.

## Why V1 Cannot Gate the Full Run

The V1 pilot completed all 32 geometry jobs, all 64 feature jobs, and all
1,320 expected metric rows. Its sentinel rerun was deterministic, no raw DINO
feature tensor was retained, and projected full-run storage was
24,622,927,703 bytes. The only failed hard gate was clean directional
coverage:

- observed clean directional coverage: `0.22916666666666666`;
- required coverage: `0.90`;
- every symmetric controlled row abstained.

The V1 pilot mixed `left__back`, which can have limited but nonzero visible
surface support, with `left__right`, whose two cameras observe opposite shoe
surfaces. Requiring 90% clean coverage over those pairs incorrectly treats
expected non-overlap as a pipeline failure.

Clean geometry support was decomposed without reading DINO scores. At 512
pixels, `left__right` had only 0–0.09% depth-consistent visible support for
the two pilot shoes. Between 94.5% and 99.2% of the projected target
foreground lay behind the visible source depth, while free-space violations
were approximately zero. This is the signature of expected self-occlusion,
not a reason to relax depth consistency.

By contrast, `right__front` had the following clean directional support:

| Shoe | Direction | 32 grid | 37 grid | 512 grid |
|---|---|---:|---:|---:|
| Air Huarache Run Ultra | right to front | 10.16% | 9.09% | 8.61% |
| Air Huarache Run Ultra | front to right | 38.46% | 27.05% | 33.77% |
| Air Max 90 Surplus | right to front | 17.42% | 20.69% | 18.14% |
| Air Max 90 Surplus | front to right | 26.83% | 30.36% | 37.47% |

At the native 32 grid, the smallest accepted direction contains 19 valid
feature tokens. This avoids the three-token support that would result from
lowering the threshold to 2% merely to retain `left__back`.

## Considered Approaches

### Option 1: Separate eligible and expected-abstention pairs

Use `right__front` as the primary pilot pair, retain `left__right` as a
negative control, lower the overlap threshold only from 10% to 8%, and gate
the two roles separately.

This is selected because it tests both successful scoring and correct
abstention without admitting extremely small native-grid support.

### Option 2: Use two adjacent pairs with a 2% threshold

This would make more adjacent pairs numerically eligible, but one observed
native direction would contain only three valid tokens. Its score would be
too sensitive to individual tokens for a resolution-fidelity pilot.

This option is rejected.

### Option 3: Keep the V1 protocol unchanged

This preserves the original threshold and pairs, but leaves the pilot with no
usable symmetric controlled score and makes the planned full analysis
unverifiable.

This option is rejected.

## Frozen V2 Design

The experiment identifier and output root become:

- experiment: `20260724_upper_v2`;
- output root:
  `/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2`;
- pilot pairs, in frozen order: `right__front`, `left__right`;
- relative depth tolerance: `0.03`;
- minimum overlap fraction: `0.08`;
- confidence percentile: `10.0`;
- physical pilot GPU: GPU 4;
- physical full-run GPUs: GPUs 4–7.

All other shoes, corruptions, backbones, feature tracks, NAF variants,
camera/depth perturbations, storage limits, and model revisions remain
unchanged from V1. The V1 run directory remains immutable evidence of the
failed protocol.

The 8% threshold is fixed from clean geometry before finite DINO score
inspection. It must not be retuned after looking at feature discrimination.

## Pilot Gate Semantics

The pilot still requires exactly 1,320 metric rows, zero failed jobs,
deterministic sentinel output, no raw feature artifacts, and projected
persistent storage no greater than 300 GiB.

Clean coverage is split by pair role:

1. `clean_primary_directional_coverage` considers only clean
   `right__front` rows. A direction is covered when its `s11` score and
   overlap are finite and overlap is at least 0.08. The required fraction is
   at least 0.90.
2. `clean_primary_pair_coverage` considers only clean `right__front` rows.
   A row is covered when the symmetric `s11` score is finite and the row did
   not abstain. The required fraction is at least 0.90.
3. `clean_negative_control_abstention` considers only clean
   `left__right` directions. A direction is correctly rejected when its
   score is non-finite or its overlap is below 0.08. The required fraction is
   at least 0.90.

For compatibility, `clean_directional_coverage` remains in the JSON report
as an alias of `clean_primary_directional_coverage`; it no longer mixes the
negative control into the denominator.

The full design continues to emit all six upper-view pairs. The 8% overlap
rule determines eligibility, and analysis continues to report abstention and
overlap rather than silently imputing missing scores. Opposite pairs are
therefore retained as diagnostic evidence but do not create artificial
feature comparisons where no visible surface correspondence exists.

## Geometry Error Versus DINO Difference

The report must keep three evidence layers separate:

1. Visibility/support diagnostics classify projected pixels as
   depth-consistent, occluded behind the source surface, or free-space
   violations. This explains whether a score is geometrically observable.
2. The controlled factorial track separates the feature effect, adaptive
   depth-geometry effect, and their interaction on common support.
3. The full geometry-calibration track applies known camera rotation,
   relative translation, depth scale, and depth noise perturbations. Its
   score-response envelope quantifies how much metric change can be
   attributed to plausible geometry error.

The cameras must always be described as
`fixed/precomputed DA3 cameras (not calibrated GT)`. The calibration track
does not turn them into ground truth; it measures sensitivity around those
fixed estimates.

## Failure Handling and Reproducibility

V2 writes to a new output root and never deletes or overwrites V1. A changed
configuration hash invalidates all V2 jobs, so the complete V2 pilot is run
from preparation through aggregation and sentinel validation.

If V2 fails a coverage role, the full run remains blocked. Thresholds or pair
roles must not be changed again without a new numbered protocol revision.
Dependency or implementation failures may be fixed with tests and a complete
pilot rerun under the same V2 scientific configuration.

## Verification

Implementation must be test-driven and verify:

- the frozen configuration contains the V2 identifier, output root, pairs,
  and 8% threshold;
- pilot design counts remain 32 states, 44 pair conditions, 64 feature jobs,
  and 1,320 metric rows;
- primary coverage ignores the negative-control pair;
- negative-control abstention is checked independently;
- a missing primary or negative-control clean subset fails the gate;
- legacy all-pair coverage cannot accidentally pass V2;
- the full test suite and `git diff --check` pass before the real pilot;
- the real V2 gate passes before an annotated V2 pilot tag is created.

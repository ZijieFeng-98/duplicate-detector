# FigCheck Integration Plan (Offline Analysis)

The environment cannot fetch the public FigCheck repository (see `external_reference/figcheck/README.md`),
so the detailed source review is blocked. To keep the integration work moving, the following plan captures
how we intend to align our pipeline with the heuristics described in FigCheck's documentation and public
presentations.

## Mapping of algorithmic components

| FigCheck Concept | Implementation Notes in This Repo |
| --- | --- |
| Lane detection for western blots | Implemented in `tools/figcheck_heuristics.py` via `_lane_mask` and `_lane_profile`, which use adaptive thresholding and morphological closing to emphasize horizontal/vertical bands before computing projections. |
| Contrast normalization | `_preprocess_gray` applies CLAHE with configurable parameters to mimic FigCheck's per-panel normalization prior to band comparison. |
| Rotation handling | `score_duplicate_figcheck_style` evaluates both 0° and 90° orientations to accommodate swapped axes or rotated crops. |
| Partial crop matching | `_partial_template_score` performs normalized template matching in both directions, covering partial overlaps analogous to FigCheck's partial duplicate scoring. |
| Composite duplicate scoring | `score_duplicate_figcheck_style` blends band alignment, projection correlation, and partial template scores with tunable weights to emulate FigCheck's band-alignment scoring heuristic. |

These hooks are guarded behind the `ENABLE_FIGCHECK_HEURISTICS` feature flag so we can roll them out
incrementally once the upstream repository is available for validation.

## Next steps once FigCheck is cloned

1. Replace the placeholder notebook logic in `comprehensive_test_scripts/notebooks/figcheck_comparison.ipynb`
   with actual fixture loading and comparative metrics.
2. Fill in the licensing compatibility section in `external_reference/figcheck/README.md`.
3. Adjust the weights and thresholds in `FigcheckHeuristicConfig` to match empirical results from
   the FigCheck benchmarks.


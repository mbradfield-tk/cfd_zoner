# CFD Zoner — Zone Creation and Results Interpretation

Companion to [usage.txt](usage.txt) (CLI reference). This document explains
*how* the zones are created and *how to read* the outputs, in particular for
answering the question: **are there significant local-vs-global differences
(gradients) in the chosen variable?**

---

## Part 1 — How zones are created

Zoning combines one **geometric zone** (the impeller region) with
**value-based zones** (k-means clustering of the field), which are then split
into **spatially contiguous sub-zones**. The pipeline runs per case and per
variable:

```
VTI field --> 1. fluid mask --> 2. impeller zone --> 3. log k-means --> 4. spatialization
               (which cells)     (geometric,          (value classes)     (contiguous
                                  always class 1)                          sub-zones)
```

### 1.1 Analysis timestep

Before zoning, the analysis time is chosen from the M-Star stats files
(out/Stats/Fluid.txt): the first time from which the windowed mean of the
variable (or power number with `--steady-on power`) stops drifting by more
than `--tol` per `--window`, persisting to the end of the trace. The first
VTI at or after that time is analyzed (`--time` overrides). time_trace.png
shows the trace with the detected steady point and the selected VTI marked.

### 1.2 Fluid mask

Selects the cells all later steps operate on:

- Cells with zero velocity magnitude are excluded (M-Star writes exact zeros
  for solid/exterior cells).
- Cells inside the moving-body STL (impeller + shaft) are excluded: unlike
  static bodies, moving-body cells carry the solid's velocity, so the
  zero-velocity blanking does not catch them.
- The fill-level STL from input.xml restricts to the liquid. Box-like meshes
  use a fast axis-aligned bounds test; arbitrary shapes use a
  point-in-surface test.

### 1.3 Impeller zone (always class 1)

The impeller region is defined **geometrically**, not by clustering, because
it is the mechanistically distinct compartment regardless of the variable:

- From the impeller STL + rotation axis (input.xml): a swept cylinder around
  the axis. Radius = max blade radius; axial span from the blade region only
  (so an included shaft does not stretch the zone); both padded by
  `--impeller-pad` (default 15%).
- Fallback without geometry: the largest connected blob of top-percentile
  velocity cells (`--impeller-percentile`), as a fitted cylinder or the raw
  blob (`--impeller-shape`).

The impeller zone is excluded from clustering and never merged. Disable it
with `--no-impeller-zone`.

### 1.4 Value clustering (classes 2+)

- The remaining fluid-cell values are transformed to log10. Turbulence
  variables (EDR, shear) span decades; linear clustering would lump
  everything below the top decade into one class (`--linear` disables this).
- 1-D k-means is fitted on a 200,000-cell random sample (fixed seed, so runs
  are reproducible) and all cells are assigned to the nearest cluster.
- Classes are ordered so **class 1 = highest values**. Class boundaries are
  the midpoints between adjacent cluster centers, mapped back to value space
  (the dashed lines on histogram.png).

#### Automatic zone count (auto-k)

When `--n-zones` is omitted, the number of value classes is chosen by a
silhouette-score sweep over k = 2 … 8 on a 20,000-cell sub-sample:

- The silhouette score (-1 … 1) measures how compact and separated the
  clusters are: ~1 means the value distribution has k genuinely distinct
  bands; ~0 means the split is arbitrary (slicing a smooth continuum).
- Each candidate is logged (`auto-k: k=3 silhouette=0.55`); the highest score
  wins, and the impeller zone is added on top (total zones = N + 1).

Notes:
- The criterion looks only at the 1-D value distribution, not at spatial
  structure. Near-log-normal fields usually peak at small k (2-3) because
  their histograms have no deep valleys.
- A clearly winning score (0.65 vs 0.4) indicates real banding; a flat score
  profile means k is arbitrary.
- In batch mode auto-k can pick different k per case, weakening the
  rank-aligned comparison — pass `--n-zones` for batch runs.

### 1.5 Spatialization (sub-zones)

Value classes are converted into physically meaningful compartments:

- Each class is split into connected components (26-neighbor connectivity):
  the same value band above vs below the impeller becomes separate sub-zones.
- Fragments smaller than `--min-zone-frac` (default 0.5% of fluid volume)
  are absorbed into their most common neighboring class, removing turbulent
  speckle so sub-zones are meaningful compartments.

Two labelings result, both written to zones.vti and reported in zones.csv:

| Label     | Meaning |
|-----------|---------|
| ZoneClass | value class (1 = impeller, 2+ = value bands high to low); rank-comparable across cases |
| ZoneID    | contiguous sub-zone (largest first) within each class |

Design idea: k-means answers *"which value band does a cell belong to?"* and
the connected-component pass answers *"where does that band form coherent
regions?"* — together giving compartments that are both value-homogeneous
and spatially contiguous, which makes the per-zone statistics and
heterogeneity metrics physically interpretable.

---

## Part 2 — How to interpret the results

### 2.1 Per-zone statistics (zones.csv)

For every class and sub-zone: cell count, volume, volume fraction, mean, std,
min, max of the variable. The class-level rows are the compartment summary;
the sub-zone rows show whether a value band is one coherent region or several
separate pockets.

### 2.2 Heterogeneity metrics (heterogeneity.csv)

Single-number answers to "are the gradients significant?". With millions of
cells classic p-values are meaningless (everything is "significant"), so
effect sizes are used:

| Metric | Question it answers | Interpretation |
|--------|--------------------|----------------|
| `eta2_between_zone` | How much of the total variability is explained by which zone a cell is in? | ~1: distinct, spatially coherent regions. ~0: field is effectively homogeneous, zoning is arbitrary. > 0.5 strong, 0.2-0.5 moderate, < 0.2 weak. |
| `zone_contrast` | How different are the extreme regions? | max/min zone mean. 1 = uniform; > ~3 usually process-relevant. Compares extremes only — check volume fractions for the volume at each extreme. |
| `cv` | How wide is the overall distribution? | std/mean over fluid cells; the classic mixing homogeneity measure (0 = homogeneous, ~1 = spread comparable to the mean). |
| `sigma_log10` | Same, for decade-spanning variables | Spread of log10(values); robust for EDR/shear. |
| `p05` … `p99` | Volume-weighted percentiles | e.g. p95 = value not exceeded in 95% of the fluid volume. |
| `p95_over_p50`, `p99_over_mean` | How extreme is the top of the distribution? | "Peakiness" relative to the bulk. |
| `grad_index` | How steep are the actual spatial gradients? | mean grad magnitude x impeller diameter / global mean. ~"the variable changes by (grad_index) x the mean per impeller diameter". Dimensionless, comparable across scales/rpm. `grad_index_p95` captures the steepest fronts. |
| `significant_gradients` | The verdict | True when eta2 >= `--eta2-threshold` (0.5) AND contrast >= `--contrast-threshold` (3.0): differences are both *organized into regions* and *large*. |

Reading them as a set:

- high eta2 + high contrast: distinct process regions — a compartment model
  is justified
- high CV + low eta2: variability is fine-grained turbulence, not regions
- high contrast + small high-zone volume: brief extreme exposure while
  material cycles (check exposure_cdf.png for the volume at risk)
- high grad_index + low eta2: steep but localized fronts (e.g. blade tips)
  in an otherwise uniform tank

The verdict criterion is a heuristic; log-scale variables like EDR often sit
near eta2 ~ 0.4-0.5 with very high contrast, which merits a human look
(tune the thresholds via CLI).

### 2.3 Diagnostic plots (per case)

| File | What it shows | How to read it |
|------|---------------|----------------|
| histogram.png | value distribution + class boundaries + zone means | Deep valleys between boundaries = natural bands; smooth continuum = zoning slices arbitrarily |
| exposure_cdf.png | cumulative fluid volume vs value | "What fraction of the batch sees <= X?"; steep curve = homogeneous, wide S = heterogeneous. Read process limits off it directly (e.g. % of volume above a shear-damage threshold) |
| profiles.png | volume-averaged axial and radial profiles | Locates the gradients: an impeller-height peak means discharge-dominated; a monotonic axial slope means top-to-bottom stratification |
| zone_slice*.png | 2D zone maps (side/top cross-sections) | Where the compartments sit; sanity-check the impeller cylinder and sub-zone splits |
| zone_pct_slice*.png | zone mean as % of global mean | Same maps normalized to the tank average (100% = average) |
| zone_weights.png | volume fraction vs contribution to the mean | A zone with 15% of volume contributing 50% of the mean is a "hot spot" driving the average |
| zones_3d.png / .html / .glb | 3D cutaway rendering | Static image, interactive browser view, and PowerPoint-insertable 3D model |

### 2.4 Batch comparison (zoner_comparison/)

Zones are clustered per case and aligned **by rank** (impeller vs impeller,
class 1 vs class 1). Absolute boundaries differ between cases; normalized
quantities (mean / case global mean) are the fair comparison.

| File | What it shows |
|------|---------------|
| comparison.csv | case x zone stats incl. zone-mean / case-global-mean ratio |
| heterogeneity_by_case.csv / heterogeneity.png | eta2, CV, contrast, gradient index per case — rank cases by non-uniformity |
| zone_ratios.png | local vs global per case (1.0 = case average) |
| zone_means.png / zone_volfrac.png / global_means.png | absolute means, volume fractions, case averages |
| zones_2d.html | zone maps per case (per-case rank colors) |
| zones_2d_common.html | zone maps on one shared discrete color scale — equal means get equal colors, so differences across cases are directly visible |
| zones_2d_normalized.html | zone means on a continuous log color scale spanning all cases |
| exposure_cdf.png | all cases' exposure curves overlaid (dashed = case means) |
| zones_3d.html | interactive 3D views, one per case, linked cameras |

### 2.5 Practical guidance

- Use **auto-k exploratively** on a single case to learn how much natural
  banding the variable has (watch the logged silhouette scores), then fix
  `--n-zones` (e.g. 4) for batch comparisons so ranks align.
- Headline answer for "significant gradients?": quote eta2 + zone contrast,
  illustrate with zone_pct_slice and exposure_cdf, and locate the gradients
  with profiles.png.
- zones.vti opens in ParaView for full 3D inspection (color by ZoneClass or
  ZoneID, threshold, slice at will).

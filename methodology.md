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

### 1.1 Analysed field: steady start and time averaging

Before zoning, the start of the steady state is found from the M-Star stats
files. Flow variables use `out/Stats/Fluid.txt` (the `velocity` alias uses
`Mean Velocity`); Scalar Field objects use the matching
`out/Stats/Scalar_<object>.txt` and its `Conc Mean` column (`--steady-on power`
uses the power number). Each sample is steady when the mean of the trailing
`--window` differs from the mean of the window before it by less than `--tol`.
Strictly, the plateau starts at the first steady sample after which every
sample is steady. Short excursions are tolerated in a plateau long enough to
judge them: an unsteady run is a series of consecutive unsteady samples; a run
spanning at least `--window` is a real change and the plateau starts after the
last such run, at the first steady sample from which at most 10 % of the samples
up to the end of the trace are unsteady and at least 10 windows of trace
remain. If neither start exists, or the plateau found is shorter than two
windows (the span the test itself compares, e.g. a quiet last second of a
fluctuating trace), no plateau is reported (the steady start stays
blank in `zoner_summary.csv` and `time_trace.csv`) and the last 25 % of the
trace's time span is used instead. `--time` sets the start time by hand.

What is analysed from that time on:

- The variable has a `Time-Avg` variant in the VTI (and `--instantaneous` is not
  given): that array is taken from the **last** frame of the series, the most
  averaged one.
- Otherwise the instantaneous field is **averaged over every frame from the
  start time to the last frame** (running sums, one frame in memory at a time).
  A cell counts as fluid when its velocity magnitude is non-zero in at least half
  of these frames; its value is the mean over the frames in which it was fluid.
- `--instantaneous`: one frame only, the first at or after the start time (the
  last frame when no plateau was found). Statistics of a single snapshot of a
  turbulent field scatter from frame to frame and are more extreme than those of
  the mean field.

`zoner_summary.csv` reports the analysis time (last frame used), the averaging
start and the number of frames averaged (1 for a single frame). `time_trace.png`
shows the trace with the detected steady point and the analysis time marked. For
scalar objects, M-Star's global mean and RSD at the analysis time are also
written to `heterogeneity.csv` and used as the case global mean in batch
comparisons.

Multi-GPU solves write one piece of the grid per GPU (`Volume/block0`, `block1`,
...). The same-named files of all blocks are assembled into one grid by their
origins; a frame that is missing in one block is an error.

### 1.2 Fluid mask

Selects the cells all later steps operate on:

- Cells with zero velocity magnitude are excluded (M-Star writes exact zeros
  for solid/exterior cells and for the gas above a free surface). Together with
  the moving-body cells these are the cells M-Star counts in `Fluid.txt` (fluid
  volume, volume means).
- Cells inside the moving body (impeller + shaft) are excluded: unlike
  static bodies, moving-body cells carry the solid's velocity, so the
  zero-velocity blanking does not catch them. The body is taken in the pose of
  the analysed frame from M-Star's posed mesh
  (`out/Output/VolumeMovingBody_<name>/block*/*.<time>.vtp`): the impeller STL,
  placed as in input.xml (`<scale>` and `<transform>` matrix), is turned about
  the rotation axis until its vertices lie on that mesh, which gives the same
  cells as the much finer posed mesh itself (used directly if no such rotation
  exists). When a field is averaged over several frames this is done per frame:
  a cell swept by a blade is not fluid in the frames in which it is inside the
  body, and its value is the mean over the other frames. Without posed-mesh
  output the placed STL in its initial pose is used.
- The fill-level STL from input.xml is **not** used when a velocity magnitude
  array is available: it is only the initial region of one fluid (the lower
  layer of a two-fluid case, the liquid at rest of a free-surface case). It
  remains the fallback for files without a usable velocity magnitude array:
  box-like meshes use a fast axis-aligned bounds test, arbitrary shapes a
  point-in-surface test. The log says which mask was used.

### 1.3 Impeller zone (always class 1)

The impeller region is defined **geometrically**, not by clustering, because
it is the mechanistically distinct compartment regardless of the variable:

- From the impeller STL + rotation axis (input.xml): a swept cylinder around
  the rotation axis, i.e. the line through the `<pivot>` of input.xml along
  `<rotationAxis>` (without a pivot: through the centre of the STL's bounding
  box, which is off the axis for e.g. three-bladed impellers). The STL is
  placed as in input.xml (`<scale>`, `<transform>` matrix). Radius = largest
  distance of the STL vertices from that line; axial span from the blade region
  only (so an included shaft does not stretch the zone); both padded by
  `--impeller-pad` (default 15%).
- Fallback without geometry: the largest connected blob of top-percentile
  velocity cells (`--impeller-percentile`), as a fitted cylinder or the raw
  blob (`--impeller-shape`).

The impeller zone is excluded from clustering, is never merged and takes no
merged fragments: it stays the geometric cylinder. Disable it with
`--no-impeller-zone`.

### 1.4 Value clustering (classes 2+)

- The remaining fluid-cell values are transformed to log10. Turbulence
  variables (EDR, shear) span decades; linear clustering would lump
  everything below the top decade into one class (`--linear` disables this).
- 1-D k-means is fitted on a 200,000-cell random sample (fixed seed, so runs
  are reproducible) and all cells are assigned to the nearest cluster.
- Classes are ordered so **class 1 = highest values**. Class boundaries are
  the midpoints between adjacent cluster centers, mapped back to value space
  (the dashed lines on histogram.png). `class_boundaries.csv` lists, per zone,
  the upper boundary of its value class: blank for the impeller zone and for
  the highest value class; with `--log-bins` it is the upper edge of the bin,
  listed only for the classes that exist after merging.

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
  are absorbed into their most common neighboring value class (never into the
  impeller zone), removing turbulent speckle so sub-zones are meaningful
  compartments.

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

Every figure's data is always written as CSV next to it (`time_trace.csv`,
`histogram.csv`, `exposure_cdf.csv`, `class_boundaries.csv`, `zone_weights.csv`,
`profiles.csv`, `zone_slices.csv` + `zone_slice_<side|side2|top>.csv` with the
ZoneClass matrices, `zoner_summary.csv` with the scalar results); the PNG/HTML
figures below are skipped with `--no-plots`. M-Star Post runs the script that way
and draws the same figures itself (Plots > Zones), for all cases of a project.

| File | What it shows | How to read it |
|------|---------------|----------------|
| histogram.png | value distribution + class boundaries + zone means | Deep valleys between boundaries = natural bands; smooth continuum = zoning slices arbitrarily |
| exposure_cdf.png | cumulative fluid volume vs value | "What fraction of the batch sees <= X?"; steep curve = homogeneous, wide S = heterogeneous. Read process limits off it directly (e.g. % of volume above a shear-damage threshold) |
| profiles.png | volume-averaged axial and radial profiles | Locates the gradients: an impeller-height peak means discharge-dominated; a monotonic axial slope means top-to-bottom stratification |
| zone_slice*.png | 2D zone maps (side/top cross-sections) | Where the compartments sit; sanity-check the impeller cylinder and sub-zone splits. The side views are cut through the impeller axis (the `<pivot>` of input.xml; the fluid centroid when there is no impeller), the top view through the impeller-zone centre; `zone_slices.csv` gives each plane's coordinate |
| zone_pct_slice*.png | zone mean as % of global mean | Same maps normalized to the tank average (100% = average) |
| zone_weights.png | volume fraction vs contribution to the mean | A zone with 15% of volume contributing 50% of the mean is a "hot spot" driving the average |
| zones_3d.png / .html / .glb | 3D cutaway rendering | Static image, interactive browser view, and PowerPoint-insertable 3D model |

### 2.4 Batch comparison (zoner_comparison/)

Zones are clustered per case and aligned **by rank** (impeller vs impeller,
class 1 vs class 1). Absolute boundaries differ between cases; normalized
quantities (mean / case global mean) are the fair comparison. (In M-Star Post the
same comparison is built from the per-case CSVs of all CFD zoner runs of a
project, newest run per case; the stand-alone batch mode writes these files.)

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

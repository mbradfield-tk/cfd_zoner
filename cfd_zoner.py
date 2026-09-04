#!/usr/bin/env python3
"""Zonal analysis of M-Star CFD volume (VTI) output.

Reads an M-Star results directory in its native layout (case/input.xml,
case/*.stl, case/out/Stats/, case/out/Output/Volume/block*/*.vti),
finds momentum steady state (or uses a user-defined time), always defines an
impeller zone, clusters the selected variable into value-based zones (log-scale
k-means) split into spatially contiguous sub-zones, and reports per-zone stats,
a labeled VTI for ParaView, diagnostic plots, and a 3D rendering.

Example:
    python cfd_zoner.py test_case_1 --variable edr --n-zones 4
    python cfd_zoner.py sim_root --batch --variable edr shear --n-zones 4
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- utilities

VAR_ALIASES = {
    "edr": "energy dissipation rate",
    "tke": "turbulent kinetic energy",
    "velocity": "velocity magnitude",
    "shear": "resolved shear stress",
    "tracer": "scalar tracer concentration",
    "kolmogorov": "kolmogorov length scale",
    "micromixing": "micromixing timescale",
}


def norm_name(s: str) -> str:
    """Lowercase and strip unit suffixes like '(W/kg)' or '[W/kg]'."""
    s = re.sub(r"[\[\(][^\]\)]*[\]\)]", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def var_unit(var_name: str) -> str:
    m = re.search(r"\(([^)]+)\)", var_name)
    return f" {m.group(1)}" if m else ""


# set False by --no-impeller-zone: class ids then start at 1 with no impeller zone
HAS_IMPELLER = True


def zone_label(c: int, class_means: dict[int, float] | None = None, unit: str = "") -> str:
    if HAS_IMPELLER:
        name = "impeller" if c == 1 else f"class {c - 1}"
    else:
        name = f"class {c}"
    if class_means and c in class_means:
        name += f" (mean {class_means[c]:.3g}{unit})"
    return name


def zone_colors(class_means: dict[int, float]) -> dict[int, tuple]:
    """Turbo colors ranked by zone mean: red = highest, blue = lowest."""
    import matplotlib.pyplot as plt

    order = sorted(class_means, key=lambda c: class_means[c])
    n = len(order)
    dim = 0.75  # darken for better contrast, esp. in 3D exports
    return {c: tuple(dim * v for v in plt.cm.turbo(0.05 + 0.9 * i / max(n - 1, 1))[:3]) + (1.0,)
            for i, c in enumerate(order)}


def var_slug(v: str) -> str:
    return re.sub(r"\W+", "_", v.strip().lower()).strip("_") or "var"


def log(msg: str) -> None:
    print(f"[cfd_zoner] {msg}")


def warn(msg: str) -> None:
    print(f"[cfd_zoner] WARNING: {msg}", file=sys.stderr)


def fail(msg: str) -> None:
    sys.exit(f"[cfd_zoner] ERROR: {msg}")


# ---------------------------------------------------------------- case discovery

@dataclass
class Case:
    root: Path
    input_xml: Path | None = None
    vti_series: list[tuple[float, Path]] = field(default_factory=list)
    stats_fluid: Path | None = None
    stats_power: Path | None = None
    stl_files: list[Path] = field(default_factory=list)


# M-Star encodes sim time in the filename, e.g. Volume.5.00000e+00.vti
TIME_RE = re.compile(r"\.([0-9]+(?:\.[0-9]+)?(?:e[+-][0-9]+)?)$", re.IGNORECASE)

# directories the tool writes itself (never treated as simulation input)
SKIP_DIRS = {"zoner_output", "zoner_comparison"}


def _rglob(root: Path, pattern: str) -> list[Path]:
    return sorted(f for f in root.rglob(pattern)
                  if not SKIP_DIRS.intersection(f.parts))


def read_pvd_times(pvd: Path) -> dict[Path, float]:
    """Map VTI paths to timesteps from a ParaView collection (.pvd) file."""
    import xml.etree.ElementTree as ET

    times: dict[Path, float] = {}
    try:
        for ds in ET.parse(pvd).getroot().iter("DataSet"):
            f, t = ds.get("file"), ds.get("timestep")
            if f and t:
                times[(pvd.parent / f).resolve()] = float(t)
    except Exception as e:
        warn(f"could not parse {pvd.name}: {e}")
    return times


def build_vti_series(vti_files: list[Path], pvd_files: list[Path]) -> list[tuple[float, Path]]:
    pvd_times: dict[Path, float] = {}
    for pvd in pvd_files:
        pvd_times.update(read_pvd_times(pvd))

    series = []
    unparsed = []
    for f in vti_files:
        m = TIME_RE.search(f.stem)
        if m:
            series.append((float(m.group(1)), f))
        elif f.resolve() in pvd_times:
            series.append((pvd_times[f.resolve()], f))
        else:
            unparsed.append(f)
    if series and unparsed:
        warn(f"ignoring {len(unparsed)} VTI file(s) without a resolvable time "
             f"(e.g. {unparsed[0].name})")
    elif not series and unparsed:
        warn("no VTI timesteps resolvable from filenames or .pvd; using file order")
        series = [(float(i), f) for i, f in enumerate(unparsed)]
    return sorted(series, key=lambda x: x[0])


def discover_case(root: Path) -> Case:
    """Discover case files, preferring M-Star's native layout:

    case/input.xml, case/*.stl, case/out/Stats/*.txt,
    case/out/Output/Volume/block*/*.vti (+ case/out/Output/*.pvd),
    with a recursive fallback for non-standard layouts.
    """
    case = Case(root=root)
    if root.is_file() and root.suffix == ".vti":
        case.vti_series = [(0.0, root)]
        return case
    if not root.is_dir():
        fail(f"case path not found: {root}")

    xml = root / "input.xml"
    hits = [xml] if xml.is_file() else _rglob(root, "input.xml")
    case.input_xml = hits[0] if hits else None

    # volume VTI series: restrict to the Volume output tree when present, so
    # slice/checkpoint VTIs in a full M-Star results directory are not swept in
    volume_dir = root / "out" / "Output" / "Volume"
    if volume_dir.is_dir():
        blocks = sorted(d for d in volume_dir.iterdir()
                        if d.is_dir() and d.name.startswith("block"))
        if len(blocks) > 1:
            warn(f"multiple volume blocks ({', '.join(b.name for b in blocks)}); "
                 f"using {blocks[0].name} only")
        vti_files = sorted((blocks[0] if blocks else volume_dir).glob("*.vti"))
        pvd_files = sorted(volume_dir.parent.glob("*.pvd"))
    else:
        vti_files = _rglob(root, "*.vti")
        pvd_files = _rglob(root, "*.pvd")
    case.vti_series = build_vti_series(vti_files, pvd_files)

    stats_dir = root / "out" / "Stats"
    stats_root = stats_dir if stats_dir.is_dir() else root
    fluid = sorted(stats_root.glob("Fluid.txt")) or _rglob(root, "Fluid.txt")
    case.stats_fluid = fluid[0] if fluid else None
    power = sorted(stats_root.glob("MovingBody_*.txt")) or _rglob(root, "MovingBody_*.txt")
    case.stats_power = power[0] if power else None

    case.stl_files = sorted(root.glob("*.stl")) or _rglob(root, "*.stl")
    return case


# ---------------------------------------------------------------- input.xml

@dataclass
class ModelInfo:
    impeller_stl: Path | None = None
    impeller_axis: np.ndarray | None = None
    impeller_diameter: float | None = None
    impeller_rpm: float | None = None
    fluid_stl: Path | None = None
    context_stls: list[Path] = field(default_factory=list)


def find_stl(name: str, stl_files: list[Path]) -> Path | None:
    for f in stl_files:
        if f.name.lower() == name.lower():
            return f
    return None


def parse_input_xml(xml_path: Path | None, stl_files: list[Path]) -> ModelInfo:
    info = ModelInfo()
    claimed: set[Path] = set()
    if xml_path is not None:
        import xml.etree.ElementTree as ET

        tree = ET.parse(xml_path)
        imp = tree.getroot().find(".//impeller")
        if imp is not None:
            stl_name = (imp.findtext("stl") or "").strip()
            info.impeller_stl = find_stl(stl_name, stl_files)
            ax = imp.findtext("rotationAxis")
            if ax:
                v = np.array([float(x) for x in ax.split()])
                info.impeller_axis = v / np.linalg.norm(v)
            d = imp.findtext("diameter")
            info.impeller_diameter = float(d) if d else None
            f = imp.findtext("freq")
            info.impeller_rpm = float(f) if f else None
        fl = tree.getroot().find(".//fluidModel/geometry/stl")
        if fl is not None and fl.text:
            info.fluid_stl = find_stl(fl.text.strip(), stl_files)
        for p in (info.impeller_stl, info.fluid_stl):
            if p is not None:
                claimed.add(p)
        log(f"input.xml parsed: impeller={info.impeller_stl.name if info.impeller_stl else None}, "
            f"axis={info.impeller_axis}, D={info.impeller_diameter}, rpm={info.impeller_rpm}, "
            f"fluid={info.fluid_stl.name if info.fluid_stl else None}")
    else:
        # filename heuristics fallback
        for f in stl_files:
            n = f.name.lower()
            if info.impeller_stl is None and ("impeller" in n or "moving" in n):
                info.impeller_stl = f
                claimed.add(f)
            elif info.fluid_stl is None and ("fluid" in n or "fill" in n):
                info.fluid_stl = f
                claimed.add(f)
    info.context_stls = [f for f in stl_files if f not in claimed]
    return info


# ---------------------------------------------------------------- stats files

def load_stats_table(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    return header, data


def find_column(header: list[str], target: str) -> int | None:
    normed = [norm_name(h) for h in header]
    if target in normed:
        return normed.index(target)
    for i, h in enumerate(normed):
        if target in h and "time-avg" not in h:
            return i
    return None


def steady_trace_from_stats(case: Case, var_query: str, steady_on: str) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Return (times, values, label) or None if no usable stats file."""
    qn = VAR_ALIASES.get(var_query.lower().strip(), norm_name(var_query))
    if steady_on == "power":
        if case.stats_power is None:
            warn("no MovingBody stats file found for power-based detection")
            return None
        header, data = load_stats_table(case.stats_power)
        col = find_column(header, "power number")
        src = case.stats_power
    else:
        if case.stats_fluid is None:
            return None
        header, data = load_stats_table(case.stats_fluid)
        col = find_column(header, f"mean {qn}")
        if col is None:
            col = find_column(header, qn)
        src = case.stats_fluid
    if col is None:
        warn(f"no matching column in {src.name}")
        return None
    log(f"steady-state trace: '{header[col]}' from {src.name}")
    return data[:, 0], data[:, col], header[col]


def detect_steady_state(times: np.ndarray, values: np.ndarray, window: float, tol: float) -> float:
    """First time from which the windowed mean stops drifting (change < tol) for good.

    Compares the mean of the trailing window against the window before it, which
    tolerates turbulent fluctuations that never settle below tol themselves.
    """
    steady_flags = np.zeros(len(times), bool)
    for i, t in enumerate(times):
        cur = (times > t - window) & (times <= t)
        prev = (times > t - 2 * window) & (times <= t - window)
        if t < times[0] + 2 * window or cur.sum() < 3 or prev.sum() < 3:
            continue
        m_cur, m_prev = values[cur].mean(), values[prev].mean()
        if abs(m_prev) > 0 and abs(m_cur - m_prev) / abs(m_prev) < tol:
            steady_flags[i] = True
    # require steadiness to persist to the end of the trace
    idx = None
    for i in range(len(times)):
        if steady_flags[i:].all() and steady_flags[i]:
            idx = i
            break
    if idx is None:
        warn(f"no steady plateau found (window={window}s, tol={tol}); using last time")
        return float(times[-1])
    return float(times[idx])


# ---------------------------------------------------------------- VTI handling

def list_vti_arrays(path: Path) -> list[str]:
    """Read array names from the XML header without loading the binary payload."""
    head = b""
    with open(path, "rb") as fh:
        while b"<AppendedData" not in head and len(head) < 4_000_000:
            chunk = fh.read(65536)
            if not chunk:
                break
            head += chunk
    return re.findall(r'DataArray[^>]*Name="([^"]+)"', head.decode("utf-8", errors="ignore"))


def resolve_variable(query: str, names: list[str], prefer_time_avg: bool) -> str:
    qn = VAR_ALIASES.get(query.lower().strip(), norm_name(query))
    normed = {n: norm_name(n) for n in names}
    want_avg = "time-avg" in qn

    def match(pool):
        exact = [n for n in pool if normed[n] == qn]
        if exact:
            return exact[0]
        sub = [n for n in pool if qn in normed[n]]
        return sub[0] if sub else None

    base_pool = names if want_avg else [n for n in names if "time-avg" not in normed[n]]
    hit = match(base_pool) or match(names)
    if hit is None:
        fail(f"variable '{query}' not found. Available: {names}")
    if prefer_time_avg and not want_avg:
        avg = [n for n in names if normed[n] == f"time-avg {normed[hit]}"]
        if avg:
            hit = avg[0]
    return hit


@dataclass
class Field3D:
    grid: "object"                 # pyvista ImageData
    nc: tuple[int, int, int]       # cells (nx, ny, nz)
    origin: np.ndarray
    spacing: np.ndarray
    var_name: str
    var: np.ndarray                # [z, y, x]
    velmag: np.ndarray | None      # [z, y, x]

    def cell_centers_1d(self):
        o, sp, nc = self.origin, self.spacing, self.nc
        return tuple(o[i] + (np.arange(nc[i], dtype=np.float32) + 0.5) * sp[i] for i in range(3))

    def cell_centers_3d(self):
        xc, yc, zc = self.cell_centers_1d()
        Z, Y, X = np.meshgrid(zc, yc, xc, indexing="ij")
        return X, Y, Z


def load_field(path: Path, var_query: str, prefer_time_avg: bool) -> Field3D:
    import pyvista as pv

    grid = pv.read(str(path))
    names = list(grid.cell_data.keys())
    scalar_names = [n for n in names if grid.cell_data[n].ndim == 1]
    var_name = resolve_variable(var_query, scalar_names, prefer_time_avg)
    log(f"analysis variable: '{var_name}'")

    nc = tuple(int(d) - 1 for d in grid.dimensions)
    shape = (nc[2], nc[1], nc[0])
    var = np.asarray(grid.cell_data[var_name], dtype=np.float32).reshape(shape)

    velmag = None
    vm = [n for n in scalar_names if norm_name(n) == "velocity magnitude"]
    if vm:
        velmag = np.asarray(grid.cell_data[vm[0]], dtype=np.float32).reshape(shape)

    return Field3D(grid=grid, nc=nc, origin=np.array(grid.origin),
                   spacing=np.array(grid.spacing), var_name=var_name,
                   var=var, velmag=velmag)


# ---------------------------------------------------------------- masks

def stl_inside_mask(fld: Field3D, stl: Path) -> np.ndarray:
    """Cells whose centers lie inside the STL surface (enclosed-points test)."""
    import pyvista as pv

    X, Y, Z = fld.cell_centers_3d()
    pts = pv.PolyData(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
    surf = pv.read(str(stl))
    try:  # pyvista >= 0.46; select_enclosed_points is deprecated
        sel = pts.select_interior_points(surf, check_surface=False)
        inside = np.asarray(sel["selected_points"])
    except AttributeError:
        sel = pts.select_enclosed_points(surf, check_surface=False)
        inside = sel["SelectedPoints"].astype(bool)
    return inside.reshape(fld.var.shape)


def build_fluid_mask(fld: Field3D, fluid_stl: Path | None,
                     moving_stl: Path | None = None) -> np.ndarray:
    # solid/exterior cells in M-Star volume output carry exact zeros
    valid = np.isfinite(fld.var)
    if fld.velmag is not None:
        valid &= fld.velmag > 0.0
        log(f"solid/exterior blanking via zero velocity magnitude: "
            f"{np.count_nonzero(~valid):,} of {valid.size:,} cells excluded")
    if moving_stl is not None:
        # moving-body cells carry the solid's velocity, so zero-blanking misses them
        solid = stl_inside_mask(fld, moving_stl)
        valid &= ~solid
        log(f"moving body '{moving_stl.name}' excluded: "
            f"{np.count_nonzero(solid):,} cells")
    if fluid_stl is None:
        return valid

    import pyvista as pv

    mesh = pv.read(str(fluid_stl))
    b = np.array(mesh.bounds).reshape(3, 2)
    boxy = False
    try:
        bounds_vol = np.prod(b[:, 1] - b[:, 0])
        boxy = bounds_vol > 0 and abs(mesh.volume - bounds_vol) / bounds_vol < 0.05
    except Exception:
        pass
    X, Y, Z = fld.cell_centers_3d()
    if boxy:
        inside = ((X >= b[0, 0]) & (X <= b[0, 1]) &
                  (Y >= b[1, 0]) & (Y <= b[1, 1]) &
                  (Z >= b[2, 0]) & (Z <= b[2, 1]))
        log(f"fluid mask from '{fluid_stl.name}' (axis-aligned box bounds)")
    else:
        inside = stl_inside_mask(fld, fluid_stl)
        log(f"fluid mask from '{fluid_stl.name}' (enclosed-points test)")
    fluid = valid & inside
    log(f"fluid cells: {np.count_nonzero(fluid):,} "
        f"({100 * np.count_nonzero(fluid) / fluid.size:.1f}% of domain)")
    return fluid


def cylinder_mask(fld: Field3D, center: np.ndarray, axis: np.ndarray,
                  radius: float, span: tuple[float, float]) -> np.ndarray:
    X, Y, Z = fld.cell_centers_3d()
    dx, dy, dz = X - center[0], Y - center[1], Z - center[2]
    axial = dx * axis[0] + dy * axis[1] + dz * axis[2]
    rad2 = ((dx - axial * axis[0]) ** 2 + (dy - axial * axis[1]) ** 2 +
            (dz - axial * axis[2]) ** 2)
    return (rad2 <= radius ** 2) & (axial >= span[0]) & (axial <= span[1])


def detect_impeller_zone(fld: Field3D, fluid: np.ndarray, model: ModelInfo,
                         pad: float, percentile: float, shape: str) -> np.ndarray:
    import pyvista as pv
    from scipy import ndimage

    if model.impeller_stl is not None and model.impeller_axis is not None:
        mesh = pv.read(str(model.impeller_stl))
        axis = np.abs(model.impeller_axis)
        center = np.array(mesh.center)
        pts = mesh.points - center
        axial = pts @ axis
        radial = np.linalg.norm(pts - np.outer(axial, axis), axis=1)
        radius = max(radial.max(), (model.impeller_diameter or 0) / 2)
        # axial span from the blade region only, so an included shaft doesn't stretch the zone
        blades = radial > 0.5 * radial.max()
        blade_axial = axial[blades] if blades.any() else axial
        radius *= 1 + pad
        span = (blade_axial.min() - pad * radius, blade_axial.max() + pad * radius)
        mask = cylinder_mask(fld, center, axis, radius, span) & fluid
        log(f"impeller zone from '{model.impeller_stl.name}': swept cylinder "
            f"R={radius * 1000:.1f} mm, axial span={span[0] * 1000:.1f}..{span[1] * 1000:.1f} mm, "
            f"{np.count_nonzero(mask):,} cells")
        return mask

    if fld.velmag is None:
        fail("no impeller STL/axis and no velocity magnitude array for fallback detection")
    warn("no impeller geometry available; falling back to velocity-maxima detection")
    thr = np.percentile(fld.velmag[fluid], percentile)
    blob = (fld.velmag >= thr) & fluid
    struct = np.ones((3, 3, 3), bool)
    lab, n = ndimage.label(blob, structure=struct)
    if n == 0:
        fail("velocity-maxima impeller detection found no cells")
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    core = lab == sizes.argmax()
    if shape == "blob":
        mask = ndimage.binary_dilation(core, structure=struct, iterations=2) & fluid
        log(f"impeller zone: dilated velocity-maxima blob, {np.count_nonzero(mask):,} cells")
        return mask
    X, Y, Z = fld.cell_centers_3d()
    axis = np.array([0.0, 1.0, 0.0])  # default tank axis when nothing else is known
    center = np.array([X[core].mean(), Y[core].mean(), Z[core].mean()])
    dxyz = np.stack([X[core] - center[0], Y[core] - center[1], Z[core] - center[2]])
    axial = axis @ dxyz
    radial = np.linalg.norm(dxyz - np.outer(axis, axial), axis=0)
    radius = radial.max() * (1 + pad)
    span = (axial.min() - pad * radius, axial.max() + pad * radius)
    mask = cylinder_mask(fld, center, axis, radius, span) & fluid
    log(f"impeller zone: cylinder fit to velocity maxima, {np.count_nonzero(mask):,} cells")
    return mask


# ---------------------------------------------------------------- zoning

def cluster_values(vals: np.ndarray, n_classes: int | None, use_log: bool,
                   log_base: float = 10.0,
                   rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (labels ordered low->high, class centers, boundaries) in value space."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if use_log:
        pos = vals[vals > 0]
        floor = pos.min() if pos.size else 1e-30
        x = (np.log(np.clip(vals, floor, None)) / np.log(log_base)).astype(np.float64)
    else:
        x = vals.astype(np.float64)
    x = x[:, None]

    rng = np.random.default_rng(rng_seed)
    fit_idx = rng.choice(len(x), min(len(x), 200_000), replace=False)

    if n_classes is None:
        best_k, best_score = None, -np.inf
        sil_idx = rng.choice(len(fit_idx), min(len(fit_idx), 20_000), replace=False)
        xs = x[fit_idx][sil_idx]
        for k in range(2, 9):
            km = KMeans(n_clusters=k, n_init=5, random_state=rng_seed).fit(xs)
            score = silhouette_score(xs, km.labels_, sample_size=min(10_000, len(xs)),
                                     random_state=rng_seed)
            log(f"  auto-k: k={k} silhouette={score:.3f}")
            if score > best_score:
                best_k, best_score = k, score
        n_classes = best_k
        log(f"auto-selected {n_classes} value classes"
            + (" (+1 impeller zone)" if HAS_IMPELLER else ""))

    if n_classes < 2:
        return np.zeros(len(vals), np.int32), np.array([x.mean()]), np.array([])

    km = KMeans(n_clusters=n_classes, n_init=10, random_state=rng_seed).fit(x[fit_idx])
    labels = km.predict(x)
    order = np.argsort(km.cluster_centers_.ravel())[::-1]  # class 1 = highest values
    remap = np.empty(n_classes, np.int32)
    remap[order] = np.arange(n_classes)
    labels = remap[labels]
    centers = km.cluster_centers_.ravel()[order]
    bounds = (centers[:-1] + centers[1:]) / 2
    if use_log:
        centers, bounds = log_base ** centers, log_base ** bounds
    return labels.astype(np.int32), centers, bounds


def bin_values(vals: np.ndarray, log_base: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classes = fixed 1-log-unit bins, e.g. decades (base 10) or doublings (base 2).

    Zone count follows the data range (unlike k-means, where the log base is
    just a rescaling and yields identical zones). Returns (labels ordered so
    0 = highest bin, bin centers, boundaries) in value space.
    """
    log_base = float(log_base)  # int base ** negative int power raises in numpy
    pos = vals[vals > 0]
    floor = pos.min() if pos.size else 1e-30
    e = np.floor(np.log(np.clip(vals, floor, None)) / np.log(log_base)).astype(np.int64)
    uniq = np.unique(e)[::-1]  # occupied bins only, highest first
    labels = np.searchsorted(-uniq, -e).astype(np.int32)
    centers = log_base ** (uniq + 0.5)
    bounds = log_base ** uniq[:-1].astype(np.float64)  # lower edge of each upper bin
    log(f"log-bin zoning (base {log_base:g}): {len(uniq)} occupied bins spanning "
        f"{log_base ** uniq[-1]:.3g} .. {log_base ** (uniq[0] + 1):.3g}")
    return labels, centers, bounds


def spatialize(zone_class: np.ndarray, fluid: np.ndarray, min_frac: float) -> np.ndarray:
    """Split classes into contiguous sub-zones; merge fragments below min_frac of fluid."""
    from scipy import ndimage

    struct = np.ones((3, 3, 3), bool)
    n_fluid = np.count_nonzero(fluid)
    min_cells = max(1, int(min_frac * n_fluid))
    first_mergeable = 2 if HAS_IMPELLER else 1  # impeller zone (1) is never merged
    classes = [c for c in np.unique(zone_class) if c >= first_mergeable]

    merged = 0
    for c in classes:
        lab, n = ndimage.label(zone_class == c, structure=struct)
        if n <= 1:
            continue
        sizes = np.bincount(lab.ravel())
        for comp, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sizes[comp] >= min_cells or sl is None:
                continue
            sle = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)
            comp_mask = lab[sle] == comp
            ring = ndimage.binary_dilation(comp_mask, structure=struct) & ~comp_mask
            neigh = zone_class[sle][ring]
            neigh = neigh[(neigh > 0) & (neigh != c)]
            if neigh.size:
                zone_class[sle][comp_mask] = np.bincount(neigh).argmax()
                merged += 1
    if merged:
        log(f"merged {merged} sub-zone fragments below {min_frac:.2%} of fluid volume")

    zone_id = np.zeros_like(zone_class)
    nid = 0
    for c in np.unique(zone_class):
        if c < 1:
            continue
        lab, n = ndimage.label(zone_class == c, structure=struct)
        sizes = np.bincount(lab.ravel())
        order = np.argsort(sizes[1:])[::-1] + 1  # biggest sub-zone first
        mapping = np.zeros(n + 1, np.int32)
        for rank, comp in enumerate(order):
            mapping[comp] = nid + rank + 1
        zone_id = np.where(lab > 0, mapping[lab], zone_id)
        nid += n
    return zone_id


# ---------------------------------------------------------------- reporting

def zone_summary(mask: np.ndarray, var: np.ndarray, cell_vol: float, n_fluid: int) -> dict:
    v = var[mask]
    return {
        "cells": int(mask.sum()),
        "volume_m3": mask.sum() * cell_vol,
        "vol_frac": mask.sum() / n_fluid,
        "mean": float(v.mean()) if v.size else np.nan,
        "std": float(v.std()) if v.size else np.nan,
        "min": float(v.min()) if v.size else np.nan,
        "max": float(v.max()) if v.size else np.nan,
    }


def collect_zone_rows(zone_class: np.ndarray, zone_id: np.ndarray, fluid: np.ndarray,
                      var: np.ndarray, cell_vol: float) -> list[tuple[str, int, str, dict]]:
    """Per-class and per-subzone stats rows: (level, id, label, summary)."""
    n_fluid = int(np.count_nonzero(fluid))
    rows = []
    for c in np.unique(zone_class):
        if c < 1:
            continue
        name = zone_label(int(c))
        rows.append(("class", int(c), name, zone_summary(zone_class == c, var, cell_vol, n_fluid)))
    for z in np.unique(zone_id):
        if z < 1:
            continue
        c = int(zone_class[zone_id == z][0])
        name = zone_label(c)
        rows.append(("subzone", int(z), name, zone_summary(zone_id == z, var, cell_vol, n_fluid)))
    return rows


def heterogeneity_stats(fld: Field3D, fluid: np.ndarray, zone_class: np.ndarray,
                        model: ModelInfo, eta2_thr: float, contrast_thr: float) -> dict:
    """Metrics quantifying whether local values differ significantly from the global mean.

    eta2 (between-zone variance fraction) ~1 means the zones capture real spatial
    gradients; ~0 means the field is essentially homogeneous. With millions of
    cells p-values are meaningless, so effect sizes are used instead.
    """
    from scipy import ndimage

    v = fld.var[fluid].astype(np.float64)
    mu = float(v.mean())
    sd = float(v.std())
    cv = sd / mu if mu else np.nan
    pos = v[v > 0]
    sigma_log10 = float(np.log10(pos).std()) if pos.size else np.nan
    pct = {f"p{q:02d}": float(np.percentile(v, q)) for q in (5, 25, 50, 75, 95, 99)}

    # variance decomposition: fraction of total variance explained by the zones
    ss_total = sd ** 2 * v.size
    ss_between = 0.0
    zmeans = []
    for c in np.unique(zone_class):
        if c < 1:
            continue
        zv = fld.var[zone_class == c]
        zm = float(zv.mean())
        ss_between += zv.size * (zm - mu) ** 2
        zmeans.append(zm)
    eta2 = ss_between / ss_total if ss_total else np.nan
    contrast = max(zmeans) / min(zmeans) if zmeans and min(zmeans) > 0 else np.nan

    # spatial gradient magnitude on interior fluid cells only: zero-filled solid
    # neighbours would fake steep gradients at the walls
    sp = fld.spacing
    gz, gy, gx = np.gradient(fld.var, sp[2], sp[1], sp[0])
    gmag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    interior = ndimage.binary_erosion(fluid, np.ones((3, 3, 3), bool))
    g = gmag[interior].astype(np.float64)
    length = model.impeller_diameter or float(np.max(fld.spacing * np.array(fld.nc)))
    grad_index = float(g.mean()) * length / mu if g.size and mu else np.nan
    grad_index_p95 = float(np.percentile(g, 95)) * length / mu if g.size and mu else np.nan

    return {
        "global_mean": mu, "cv": cv, "sigma_log10": sigma_log10, **pct,
        "p95_over_p50": pct["p95"] / pct["p50"] if pct["p50"] else np.nan,
        "p99_over_mean": pct["p99"] / mu if mu else np.nan,
        "eta2_between_zone": eta2, "zone_contrast": contrast,
        "grad_index": grad_index, "grad_index_p95": grad_index_p95,
        "significant_gradients": bool(eta2 >= eta2_thr and contrast >= contrast_thr),
    }


def report_heterogeneity(het: dict, var_name: str, out_csv: Path,
                         eta2_thr: float, contrast_thr: float) -> None:
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        for k, v in het.items():
            w.writerow([k, v])
    log(f"wrote {out_csv}")

    verdict = "SIGNIFICANT" if het["significant_gradients"] else "not significant"
    print(f"\nHeterogeneity of '{var_name}':")
    print(f"  CV = {het['cv']:.3f}   sigma(log10) = {het['sigma_log10']:.3f}   "
          f"P95/P50 = {het['p95_over_p50']:.2f}   P99/mean = {het['p99_over_mean']:.2f}")
    print(f"  between-zone variance fraction eta^2 = {het['eta2_between_zone']:.3f}   "
          f"zone contrast = {het['zone_contrast']:.2f}   "
          f"gradient index = {het['grad_index']:.3f}")
    print(f"  => local-vs-global gradients {verdict} "
          f"(criteria: eta^2 >= {eta2_thr} and zone contrast >= {contrast_thr})")


@dataclass
class CaseResult:
    """Class-level zoning results of one case/variable, for cross-case comparison."""
    case_name: str
    var_name: str
    t_sel: float
    global_mean: float
    class_rows: list[tuple[int, str, dict]]  # (class id, zone label, stats summary)
    class_means: dict[int, float]
    out_dir: Path                            # holds zones.vti for comparison visuals
    model: ModelInfo
    het: dict = field(default_factory=dict)  # heterogeneity_stats() output


def report(rows: list[tuple[str, int, str, dict]], var_name: str, out_csv: Path) -> None:
    hdr = f"{'level':8} {'id':>3} {'zone':12} {'cells':>10} {'vol [m3]':>12} {'vol %':>7} " \
          f"{'mean':>12} {'std':>12} {'min':>12} {'max':>12}"
    print(f"\nZonal statistics for '{var_name}':")
    print(hdr)
    print("-" * len(hdr))
    for level, zid, name, s in rows:
        print(f"{level:8} {zid:>3} {name:12} {s['cells']:>10,} {s['volume_m3']:>12.4e} "
              f"{100 * s['vol_frac']:>6.2f}% {s['mean']:>12.4e} {s['std']:>12.4e} "
              f"{s['min']:>12.4e} {s['max']:>12.4e}")

    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["level", "id", "zone", "cells", "volume_m3", "volume_fraction",
                    f"mean {var_name}", "std", "min", "max"])
        for level, zid, name, s in rows:
            w.writerow([level, zid, name, s["cells"], s["volume_m3"], s["vol_frac"],
                        s["mean"], s["std"], s["min"], s["max"]])
    log(f"wrote {out_csv}")


def write_labeled_vti(fld: Field3D, zone_class: np.ndarray, zone_id: np.ndarray,
                      out_path: Path) -> None:
    import pyvista as pv

    out = pv.ImageData(dimensions=fld.grid.dimensions, origin=fld.grid.origin,
                       spacing=fld.grid.spacing)
    out.cell_data["ZoneClass"] = zone_class.ravel()
    out.cell_data["ZoneID"] = zone_id.ravel()
    out.cell_data[fld.var_name] = fld.var.ravel()
    out.save(str(out_path))
    log(f"wrote {out_path}")


# ---------------------------------------------------------------- plots & render

def make_plots(out_dir: Path, fld: Field3D, fluid: np.ndarray, zone_class: np.ndarray,
               bounds: np.ndarray, trace: tuple | None, t_steady: float | None,
               t_sel: float, axis: np.ndarray | None,
               class_means: dict[int, float] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if trace is not None:
        times, values, label = trace
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(times, values, lw=1)
        if t_steady is not None:
            ax.axvline(t_steady, color="tab:green", ls="--", label=f"steady @ {t_steady:.2f} s")
        ax.axvline(t_sel, color="tab:red", ls=":", label=f"analysis VTI @ {t_sel:.2f} s")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(label)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "time_trace.png", dpi=150)
        plt.close(fig)

    unit = var_unit(fld.var_name)
    colors = zone_colors(class_means) if class_means else {}
    vals = fld.var[fluid]
    pos = vals[vals > 0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if pos.size:
        ax.hist(pos, bins=np.logspace(np.log10(pos.min()), np.log10(pos.max()), 120),
                color="steelblue", alpha=0.8)
        ax.set_xscale("log")
    for i, b in enumerate(bounds):
        ax.axvline(b, color="k", ls="--", lw=1, label="class boundary" if i == 0 else None)
    if class_means:
        for c in sorted(class_means):
            ax.axvline(class_means[c], color=colors[c], ls=":",
                       lw=1.5, label=zone_label(c, class_means, unit))
    ax.legend(fontsize=8)
    ax.set_xlabel(fld.var_name)
    ax.set_ylabel("cell count")
    ax.set_title("Distribution with zone class boundaries and zone means")
    fig.tight_layout()
    fig.savefig(out_dir / "histogram.png", dpi=150)
    plt.close(fig)

    # volume fraction vs. contribution of each zone to the overall mean
    if class_means:
        cls = sorted(class_means)
        n_fluid = np.count_nonzero(fluid)
        vfrac = np.array([np.count_nonzero(zone_class == c) / n_fluid for c in cls])
        weight = vfrac * np.array([class_means[c] for c in cls])
        weight /= weight.sum()  # overall mean = sum(vol_frac * zone mean)
        xpos = np.arange(len(cls))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, c in enumerate(cls):
            ax.bar(xpos[i] - 0.2, 100 * vfrac[i], width=0.4, color=colors[c])
            ax.bar(xpos[i] + 0.2, 100 * weight[i], width=0.4, color=colors[c],
                   hatch="//", edgecolor="k", lw=0.5)
            ax.text(xpos[i] - 0.2, 100 * vfrac[i] + 1, f"{100 * vfrac[i]:.1f}%",
                    ha="center", fontsize=8)
            ax.text(xpos[i] + 0.2, 100 * weight[i] + 1, f"{100 * weight[i]:.1f}%",
                    ha="center", fontsize=8)
        ax.bar(np.nan, np.nan, color="gray", label="volume fraction")
        ax.bar(np.nan, np.nan, color="gray", hatch="//", edgecolor="k", lw=0.5,
               label="contribution to overall mean")
        ax.set_xticks(xpos)
        ax.set_xticklabels([zone_label(c, class_means, unit) for c in cls],
                           rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("percent [%]")
        overall = float((vfrac * np.array([class_means[c] for c in cls])).sum())
        ax.set_title(f"Zone volume fractions and weighting of overall mean "
                     f"({overall:.3g}{unit})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "zone_weights.png", dpi=150)
        plt.close(fig)

    # vertical cross-section containing the rotation axis
    u = np.abs(axis) if axis is not None else np.array([0.0, 1.0, 0.0])
    xc, yc, zc = fld.cell_centers_1d()
    n_classes = int(zone_class.max())
    cmap = ListedColormap([colors.get(c, plt.cm.turbo(0.5)) for c in range(1, n_classes + 1)])

    # fixed rects: varying colorbar label widths must not resize the map
    map_rect, cbar_rect = [0.09, 0.07, 0.60, 0.86], [0.71, 0.07, 0.03, 0.86]

    def plot_slice(img, extent, xlabel, ylabel, title, fname):
        img = img.astype(float)
        img[img < 1] = np.nan
        fig = plt.figure(figsize=(8, 7))
        ax2 = fig.add_axes(map_rect)
        cax = fig.add_axes(cbar_rect)
        im = ax2.imshow(img, origin="lower", extent=extent, cmap=cmap,
                        vmin=0.5, vmax=n_classes + 0.5, interpolation="nearest")
        cbar = fig.colorbar(im, cax=cax, ticks=range(1, n_classes + 1))
        cbar.ax.set_yticklabels([zone_label(c, class_means, unit)
                                 for c in range(1, n_classes + 1)], fontsize=8)
        cbar.set_label("zone (class 1 = highest values)")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_title(title)
        ax2.set_aspect("equal")
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    # per-cell lookup: zone mean as % of the global (fluid) mean
    overall_mean = float(fld.var[fluid].mean())
    pct_lut = np.full(n_classes + 1, np.nan)
    if class_means:
        for c, m in class_means.items():
            pct_lut[c] = 100 * m / overall_mean

    def plot_pct_slice(img_cls, extent, xlabel, ylabel, title, fname):
        img = np.where(img_cls >= 1, pct_lut[np.clip(img_cls, 0, n_classes)], np.nan)
        fig = plt.figure(figsize=(8, 7))
        ax2 = fig.add_axes(map_rect)
        cax = fig.add_axes(cbar_rect)
        im = ax2.imshow(img, origin="lower", extent=extent, cmap="turbo",
                        vmin=0, vmax=np.nanmax(pct_lut), interpolation="nearest")
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"zone mean [% of global mean {overall_mean:.3g}{unit}]")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_title(title)
        ax2.set_aspect("equal")
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    # slice through the tank axis (coordinate 0), not the grid mid-index: the
    # domain is not necessarily centered, so mid-index planes can cut off-axis
    kx, ky, kz = (int(np.argmin(np.abs(cc))) for cc in (xc, yc, zc))
    if u[2] < 0.9:  # slice normal to z, plus perpendicular side view normal to x
        views = [
            (zone_class[kz, :, :], [xc[0], xc[-1], yc[0], yc[-1]],
             "x [m]", "y [m]", "side view, facing +Z", ""),
            (zone_class[:, :, kx].T, [zc[0], zc[-1], yc[0], yc[-1]],
             "z [m]", "y [m]", "side view, facing +X", "_side2"),
        ]
    else:  # z-axis tank: slice normal to y, plus perpendicular side view normal to x
        views = [
            (zone_class[:, ky, :], [xc[0], xc[-1], zc[0], zc[-1]],
             "x [m]", "z [m]", "side view, facing +Y", ""),
            (zone_class[:, :, kx], [yc[0], yc[-1], zc[0], zc[-1]],
             "y [m]", "z [m]", "side view, facing +X", "_side2"),
        ]

    # top view: slice normal to the shaft axis through the impeller-zone centre
    ax_xyz = int(np.argmax(u))
    np_ax = 2 - ax_xyz  # zone_class axes are [z, y, x]
    imp_idx = np.nonzero(zone_class == 1)[np_ax]
    k = int(round(imp_idx.mean())) if imp_idx.size else zone_class.shape[np_ax] // 2
    if ax_xyz == 1:
        views.append((zone_class[:, k, :], [xc[0], xc[-1], zc[0], zc[-1]], "x [m]", "z [m]",
                      f"top view, y = {yc[k]:.4f} m", "_top"))
    elif ax_xyz == 2:
        views.append((zone_class[k, :, :], [xc[0], xc[-1], yc[0], yc[-1]], "x [m]", "y [m]",
                      f"top view, z = {zc[k]:.4f} m", "_top"))
    else:
        views.append((zone_class[:, :, k], [yc[0], yc[-1], zc[0], zc[-1]], "y [m]", "z [m]",
                      f"top view, x = {xc[k]:.4f} m", "_top"))

    for img, extent, xl, yl, desc, suf in views:
        plot_slice(img, extent, xl, yl, f"Zone map ({desc})", f"zone_slice{suf}.png")
        if class_means:
            plot_pct_slice(img, extent, xl, yl,
                           f"Zone mean as % of global mean ({desc})",
                           f"zone_pct_slice{suf}.png")

    # volume-exposure curve: what fraction of the batch sees <= a given value
    sv = np.sort(vals.astype(np.float64))
    idx = np.linspace(0, sv.size - 1, min(4000, sv.size)).astype(int)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sv[idx], 100 * (idx + 1) / sv.size, lw=1.5, color="steelblue")
    if (sv > 0).any():
        ax.set_xscale("log")
    ax.axvline(overall_mean, color="k", ls="--", lw=1,
               label=f"global mean ({overall_mean:.3g}{unit})")
    if class_means:
        for c in sorted(class_means):
            ax.axvline(class_means[c], color=colors[c], ls=":", lw=1.5,
                       label=zone_label(c, class_means, unit))
    ax.set_xlabel(fld.var_name)
    ax.set_ylabel("cumulative fluid volume [%]")
    ax.set_title("Volume exposure: fraction of fluid at or below a value")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "exposure_cdf.png", dpi=150)
    plt.close(fig)

    # axial & radial profiles: where the gradients sit macroscopically
    X, Y, Z = fld.cell_centers_3d()
    up_i = int(np.argmax(u))
    coords = (X, Y, Z)
    a = coords[up_i][fluid].astype(np.float64)
    others = [coords[i][fluid].astype(np.float64) for i in range(3) if i != up_i]
    r = np.hypot(others[0] - others[0].mean(), others[1] - others[1].mean())
    vv = fld.var[fluid].astype(np.float64)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for axp, coord, xlab in ((ax1, a, f"{'xyz'[up_i]} [m] (along shaft axis)"),
                             (ax2, r, "radius from tank axis [m]")):
        edges = np.linspace(coord.min(), coord.max(), 41)
        sums, _ = np.histogram(coord, edges, weights=vv)
        cnts, _ = np.histogram(coord, edges)
        prof = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
        axp.plot((edges[:-1] + edges[1:]) / 2, prof, lw=1.5, color="steelblue")
        axp.axhline(overall_mean, color="k", ls="--", lw=1, label="global mean")
        if (vv > 0).any():
            axp.set_yscale("log")
        axp.set_xlabel(xlab)
        axp.set_ylabel(fld.var_name)
        axp.legend(fontsize=8)
    fig.suptitle("Volume-averaged profiles")
    fig.tight_layout()
    fig.savefig(out_dir / "profiles.png", dpi=150)
    plt.close(fig)
    log(f"wrote plots to {out_dir}")


def export_glb(pl, out_path: Path, up: np.ndarray | None = None) -> None:
    """Write a binary glTF (.glb) with the model's up axis mapped to glTF +Y.

    VTK's exporter only emits JSON .gltf and bakes in a Z-up assumption, which
    misorients Y-up tanks (wrong turntable axis in PowerPoint).
    """
    import base64
    import json
    import struct
    import tempfile

    # rotation taking the scene up-vector onto glTF's +Y (Rodrigues)
    u = np.array([0.0, 1.0, 0.0]) if up is None else np.asarray(up, float)
    u = u / np.linalg.norm(u)
    y = np.array([0.0, 1.0, 0.0])
    v, c = np.cross(u, y), float(u @ y)
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        rot = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        k = v / s
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        rot = np.eye(3) + s * K + (1 - c) * (K @ K)
    m = np.eye(4)
    m[:3, :3] = rot
    node_matrix = m.T.ravel().tolist()  # glTF matrices are column-major

    with tempfile.TemporaryDirectory() as td:
        gltf = Path(td) / "scene.gltf"
        pl.export_gltf(str(gltf), inline_data=True)
        doc = json.loads(gltf.read_text())
        blob = bytearray()
        offsets = []
        for buf in doc.get("buffers", []):
            blob += b"\0" * ((-len(blob)) % 4)  # glTF requires 4-byte alignment
            offsets.append(len(blob))
            uri = buf.get("uri", "")
            if uri.startswith("data:"):
                blob += base64.b64decode(uri.split(",", 1)[1])
            elif uri:
                blob += (gltf.parent / uri).read_bytes()

    for bv in doc.get("bufferViews", []):
        bv["byteOffset"] = bv.get("byteOffset", 0) + offsets[bv.get("buffer", 0)]
        bv["buffer"] = 0
    doc["buffers"] = [{"byteLength": len(blob)}]

    # override the exporter's per-mesh transforms and drop the camera node
    doc.pop("cameras", None)
    for node in doc.get("nodes", []):
        node.pop("camera", None)
        if "mesh" in node:
            node["matrix"] = node_matrix
        elif "children" not in node:
            node.pop("matrix", None)

    # glTF defaults to alphaMode OPAQUE, which ignores baseColorFactor alpha
    for mat in doc.get("materials", []):
        rgba = mat.get("pbrMetallicRoughness", {}).get("baseColorFactor")
        if rgba and len(rgba) == 4 and rgba[3] < 1.0:
            mat["alphaMode"] = "BLEND"
            mat["doubleSided"] = True

    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    blob += b"\0" * ((-len(blob)) % 4)
    with open(out_path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fh.write(struct.pack("<II", len(blob), 0x004E4942) + blob)


def render_3d(out_dir: Path, fld: Field3D, zone_class: np.ndarray, model: ModelInfo,
              show: bool, html: bool, cutaway: bool = True,
              class_means: dict[int, float] | None = None) -> None:
    import matplotlib.pyplot as plt
    import pyvista as pv

    lab = pv.ImageData(dimensions=fld.grid.dimensions, origin=fld.grid.origin,
                       spacing=fld.grid.spacing)
    lab.cell_data["ZoneClass"] = zone_class.ravel()
    center = lab.center

    pl = pv.Plotter(off_screen=not show, window_size=(1400, 1000))
    n_classes = int(zone_class.max())
    rank_by_class = class_means or {c: float(n_classes - c) for c in range(1, n_classes + 1)}
    colors = zone_colors(rank_by_class)
    for c in range(1, n_classes + 1):
        zone = lab.threshold([c - 0.5, c + 0.5], scalars="ZoneClass")
        if cutaway and zone.n_cells:
            zone = zone.clip(normal="z", origin=center, invert=False)
        if zone.n_cells == 0:  # zone may vanish entirely in the cutaway
            continue
        name = zone_label(c, class_means, var_unit(fld.var_name))
        pl.add_mesh(zone, color=colors[c][:3],
                    opacity=0.95 if c == 1 else 0.6, label=name)
    for stl in model.context_stls:
        try:
            mesh = pv.read(str(stl))
            translucent = "vessel" in stl.name.lower() or "viewer" in stl.name.lower()
            pl.add_mesh(mesh, color="lightgray", opacity=0.08 if translucent else 0.4)
        except Exception:
            pass
    if model.impeller_stl is not None:
        pl.add_mesh(pv.read(str(model.impeller_stl)), color="dimgray")
    pl.add_legend(bcolor="white", size=(0.32, 0.18), loc="upper left")
    up = tuple(np.abs(model.impeller_axis)) if model.impeller_axis is not None else (0, 1, 0)
    pl.view_vector((1.0, 0.6, 1.0), viewup=up)
    pl.reset_camera(bounds=lab.bounds)  # keep focus on the tank, not far-reaching probe/shaft STLs
    pl.camera.zoom(0.85)

    try:
        if html:
            pl.export_html(str(out_dir / "zones_3d.html"))
            log(f"wrote {out_dir / 'zones_3d.html'}")
            # binary glTF: insertable into PowerPoint as a 3D model
            export_glb(pl, out_dir / "zones_3d.glb", up=np.array(up, float))
            log(f"wrote {out_dir / 'zones_3d.glb'}")
        if show:
            pl.show(screenshot=str(out_dir / "zones_3d.png"))
        else:
            pl.screenshot(str(out_dir / "zones_3d.png"))
        log(f"wrote {out_dir / 'zones_3d.png'}")
    except Exception as e:  # headless rendering can fail without a display/GPU
        warn(f"3D rendering failed ({e}); labeled VTI can be viewed in ParaView instead")


# ---------------------------------------------------------------- pipeline

def run_case(case: Case, model: ModelInfo, variable: str, args, out_dir: Path) -> CaseResult:
    """Full zoning/analysis pipeline for one case and one variable."""
    trace = steady_trace_from_stats(case, variable, args.steady_on)
    t_steady = None
    if args.time is not None:
        t_target = args.time
        log(f"user-defined analysis time: {t_target} s")
    else:
        if trace is None:
            warn("no stats trace available; using last VTI timestep")
            t_target = case.vti_series[-1][0]
        else:
            t_steady = detect_steady_state(trace[0], trace[1], args.window, args.tol)
            t_target = t_steady
            log(f"steady state detected at t = {t_steady:.3f} s")
    later = [(t, f) for t, f in case.vti_series if t >= t_target - 1e-9]
    t_sel, vti_path = later[0] if later else case.vti_series[-1]
    log(f"analysis timestep: t = {t_sel:.3g} s ({vti_path.name})")

    # --- load field & masks
    fld = load_field(vti_path, variable, prefer_time_avg=not args.instantaneous)
    fluid = build_fluid_mask(fld, model.fluid_stl, model.impeller_stl)
    if not fluid.any():
        fail("fluid mask is empty")

    if HAS_IMPELLER:
        impeller = detect_impeller_zone(fld, fluid, model, args.impeller_pad,
                                        args.impeller_percentile, args.impeller_shape)
    else:
        impeller = np.zeros(fld.var.shape, bool)
        log("impeller zone disabled; all zones from clustering")

    # --- cluster the remainder
    n_classes = None
    if args.n_zones is not None:
        n_classes = args.n_zones - 1 if HAS_IMPELLER else args.n_zones
    rest = fluid & ~impeller
    if args.log_bins:
        if args.n_zones is not None:
            warn("--n-zones is ignored with --log-bins (zone count follows the data range)")
        labels, centers, bounds = bin_values(fld.var[rest], args.log_base)
    else:
        labels, centers, bounds = cluster_values(fld.var[rest], n_classes,
                                                 use_log=not args.linear,
                                                 log_base=args.log_base)

    class_offset = 2 if HAS_IMPELLER else 1
    zone_class = np.zeros(fld.var.shape, np.int32)
    zone_class[impeller] = 1
    zone_class[rest] = labels + class_offset
    zone_id = spatialize(zone_class, fluid, args.min_zone_frac)

    # --- outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_vol = float(np.prod(fld.spacing))
    class_means = {int(c): float(fld.var[zone_class == c].mean())
                   for c in np.unique(zone_class) if c >= 1}
    rows = collect_zone_rows(zone_class, zone_id, fluid, fld.var, cell_vol)
    report(rows, fld.var_name, out_dir / "zones.csv")
    het = heterogeneity_stats(fld, fluid, zone_class, model,
                              args.eta2_threshold, args.contrast_threshold)
    report_heterogeneity(het, fld.var_name, out_dir / "heterogeneity.csv",
                         args.eta2_threshold, args.contrast_threshold)
    write_labeled_vti(fld, zone_class, zone_id, out_dir / "zones.vti")
    if not args.no_plots:
        make_plots(out_dir, fld, fluid, zone_class, bounds, trace, t_steady, t_sel,
                   model.impeller_axis, class_means)
    if not args.no_render:
        render_3d(out_dir, fld, zone_class, model, show=args.show, html=args.html,
                  class_means=class_means)
    return CaseResult(case_name=case.root.name, var_name=fld.var_name, t_sel=t_sel,
                      global_mean=float(fld.var[fluid].mean()),
                      class_rows=[(c, name, s) for level, c, name, s in rows
                                  if level == "class"],
                      class_means=class_means, out_dir=out_dir, model=model, het=het)


# ---------------------------------------------------------------- batch comparison

def compare_cases(results: list[CaseResult], out_dir: Path) -> None:
    """Cross-case comparison of rank-aligned zones for one variable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    var_name = results[0].var_name
    unit = var_unit(var_name)
    class_ids = sorted({c for r in results for c, _, _ in r.class_rows})
    counts = {r.case_name: len(r.class_rows) for r in results}
    if len(set(counts.values())) > 1:
        warn(f"zone counts differ across cases {counts}; ranks aligned where present")
    labels = {c: zone_label(c) for c in class_ids}

    csv_path = out_dir / "comparison.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "zone", "class", "cells", "volume_m3", "volume_fraction",
                    f"mean {var_name}", "std", "min", "max",
                    "case global mean", "mean / global mean"])
        for r in results:
            for c, name, s in r.class_rows:
                ratio = s["mean"] / r.global_mean if r.global_mean else np.nan
                w.writerow([r.case_name, name, c, s["cells"], s["volume_m3"],
                            s["vol_frac"], s["mean"], s["std"], s["min"], s["max"],
                            r.global_mean, ratio])
    log(f"wrote {csv_path}")

    x = np.arange(len(class_ids))
    n_cases = len(results)
    width = 0.8 / n_cases

    def grouped_bars(metric, fname, ylabel, title, logy=False, refline=None):
        fig, ax = plt.subplots(figsize=(max(8.0, 1.5 + 0.9 * len(class_ids) * n_cases), 4.8))
        for j, r in enumerate(results):
            by_c = {c: s for c, _, s in r.class_rows}
            vals = [metric(by_c[c], r) if c in by_c else np.nan for c in class_ids]
            ax.bar(x + (j - (n_cases - 1) / 2) * width, vals, width, label=r.case_name)
        if refline is not None:
            ax.axhline(refline, color="k", ls="--", lw=1)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([labels[c] for c in class_ids])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    grouped_bars(lambda s, r: s["mean"], "zone_means.png",
                 f"zone mean{unit}",
                 f"Zone means of {var_name} (per-case zoning, rank-aligned)", logy=True)
    grouped_bars(lambda s, r: s["mean"] / r.global_mean, "zone_ratios.png",
                 "zone mean / case global mean",
                 f"Local vs global {var_name} (1.0 = case average)", refline=1.0)
    grouped_bars(lambda s, r: 100 * s["vol_frac"], "zone_volfrac.png",
                 "volume fraction [%]", "Zone volume fractions by case")

    fig, ax = plt.subplots(figsize=(max(6.0, 1.5 + 1.2 * n_cases), 4.8))
    ax.bar([r.case_name for r in results], [r.global_mean for r in results],
           color="steelblue")
    ax.set_ylabel(f"global fluid mean{unit}")
    ax.set_title(f"Case global means of {var_name}")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_dir / "global_means.png", dpi=150)
    plt.close(fig)

    # heterogeneity indices by case
    if all(r.het for r in results):
        keys = list(results[0].het.keys())
        with open(out_dir / "heterogeneity_by_case.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["case"] + keys)
            for r in results:
                w.writerow([r.case_name] + [r.het.get(k) for k in keys])
        log(f"wrote {out_dir / 'heterogeneity_by_case.csv'}")

        metrics = [("eta2_between_zone", "between-zone variance fraction (eta^2)"),
                   ("cv", "coefficient of variation (std/mean)"),
                   ("zone_contrast", "zone contrast (max/min zone mean)"),
                   ("grad_index", "normalized gradient index")]
        names = [r.case_name for r in results]
        fig, axes = plt.subplots(2, 2, figsize=(max(10.0, 2.5 * n_cases), 8))
        for axp, (k, title) in zip(axes.ravel(), metrics):
            axp.bar(names, [r.het.get(k, np.nan) for r in results], color="steelblue")
            axp.set_title(title, fontsize=10)
            axp.tick_params(axis="x", rotation=15, labelsize=8)
        fig.suptitle(f"Heterogeneity indices \u2014 {var_name}")
        fig.tight_layout()
        fig.savefig(out_dir / "heterogeneity.png", dpi=150)
        plt.close(fig)

        print(f"\nHeterogeneity by case for '{var_name}':")
        hhdr = f"{'case':32} {'eta^2':>8} {'CV':>8} {'contrast':>9} {'grad idx':>9}  verdict"
        print(hhdr)
        print("-" * len(hhdr))
        for r in results:
            h = r.het
            verdict = "SIGNIFICANT" if h.get("significant_gradients") else "not significant"
            print(f"{r.case_name:32} {h.get('eta2_between_zone', np.nan):>8.3f} "
                  f"{h.get('cv', np.nan):>8.3f} {h.get('zone_contrast', np.nan):>9.2f} "
                  f"{h.get('grad_index', np.nan):>9.3f}  {verdict}")
    log(f"wrote comparison plots to {out_dir}")

    print(f"\nZone mean / case global mean for '{var_name}':")
    hdr = f"{'case':32} " + "".join(f"{labels[c]:>12}" for c in class_ids)
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        by_c = {c: s for c, _, s in r.class_rows}
        cells = "".join(f"{by_c[c]['mean'] / r.global_mean:>12.3f}" if c in by_c
                        else f"{'-':>12}" for c in class_ids)
        print(f"{r.case_name:32} {cells}")


def load_zone_class(result: CaseResult):
    """Reload the labeled grid written by run_case for comparison visuals."""
    import pyvista as pv

    grid = pv.read(str(result.out_dir / "zones.vti"))
    nc = tuple(int(d) - 1 for d in grid.dimensions)
    zone_class = np.asarray(grid.cell_data["ZoneClass"]).reshape((nc[2], nc[1], nc[0]))
    return grid, zone_class, nc


def zone_slice_2d(r: CaseResult):
    """Vertical cross-section of ZoneClass through the tank axis:
    (int image, extent, xlabel, ylabel)."""
    grid, zone_class, nc = load_zone_class(r)
    o, sp = np.array(grid.origin), np.array(grid.spacing)
    xc, yc, zc = (o[j] + (np.arange(nc[j]) + 0.5) * sp[j] for j in range(3))
    u = (np.abs(r.model.impeller_axis) if r.model.impeller_axis is not None
         else np.array([0.0, 1.0, 0.0]))
    if u[2] < 0.9:  # y-up tank: slice normal to z through z=0 (not the grid mid-index)
        k = int(np.argmin(np.abs(zc)))
        img = zone_class[k, :, :]
        extent = [xc[0], xc[-1], yc[0], yc[-1]]
        xl, yl = "x [m]", "y [m]"
    else:  # z-up tank: slice normal to y through y=0
        k = int(np.argmin(np.abs(yc)))
        img = zone_class[:, k, :]
        extent = [xc[0], xc[-1], zc[0], zc[-1]]
        xl, yl = "x [m]", "z [m]"
    return img, extent, xl, yl


def fig_to_html(fig, title: str, out_path: Path) -> None:
    """Save a matplotlib figure as a self-contained HTML page."""
    import base64
    import io
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    out_path.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title></head>"
        "<body style='margin:0;background:#fff'>"
        f"<img style='width:100%' src='data:image/png;base64,{b64}'/></body></html>")
    log(f"wrote {out_path}")


def _case_grid(n: int):
    ncols = min(3, n)
    return ncols, -(-n // ncols)


def _shared_extent(slices) -> tuple[float, float, float, float]:
    """Union of per-case slice extents so all subplots render at one scale."""
    ex = [e for _, e, _, _ in slices]
    return (min(e[0] for e in ex), max(e[1] for e in ex),
            min(e[2] for e in ex), max(e[3] for e in ex))


def _slice_data_bounds(img, extent, margin: float = 0.03):
    """Bounding box of labeled (>= 1) cells in slice coordinates, so a subplot
    can be cropped to the reactor instead of the full simulation domain."""
    ii, jj = np.nonzero(img >= 1)
    if ii.size == 0:
        return extent[0], extent[1], extent[2], extent[3]
    ny, nx = img.shape
    dx = (extent[1] - extent[0]) / nx
    dy = (extent[3] - extent[2]) / ny
    x0, x1 = extent[0] + jj.min() * dx, extent[0] + (jj.max() + 1) * dx
    y0, y1 = extent[2] + ii.min() * dy, extent[2] + (ii.max() + 1) * dy
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    return x0 - mx, x1 + mx, y0 - my, y1 + my


def _grouped_limits(slices) -> tuple[list[tuple[float, float, float, float]], float]:
    """Per-case axis limits and a common box aspect: same-diameter vessels
    (5 mm bins on slice width) share identical limits with the diameter spanning
    the full subplot width, so all reactors render at the same displayed size."""
    case_bounds = [_slice_data_bounds(img, ext) for img, ext, _, _ in slices]
    gkey = [round((b[1] - b[0]) / 0.005) for b in case_bounds]
    glim: dict[int, tuple[float, float, float, float]] = {}
    for k, b in zip(gkey, case_bounds):
        g = glim.get(k)
        glim[k] = b if g is None else (min(g[0], b[0]), max(g[1], b[1]),
                                       min(g[2], b[2]), max(g[3], b[3]))
    box_aspect = max((y1 - y0) / (x1 - x0) for x0, x1, y0, y1 in glim.values())
    # y-span tied to the x-span keeps meters undistorted within each subplot
    limits = []
    for k in gkey:
        x0, x1, y0, _ = glim[k]
        limits.append((x0, x1, y0, y0 + (x1 - x0) * box_aspect))
    return limits, box_aspect


def batch_html_2d(results: list[CaseResult], out_path: Path) -> None:
    """Single HTML with a grid of 2D side-view zone maps, one subplot per case."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    n = len(results)
    ncols, nrows = _case_grid(n)
    slices = [zone_slice_2d(r) for r in results]
    x0, x1, y0, y1 = _shared_extent(slices)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for i, r in enumerate(results):
        ax = axes.ravel()[i]
        img, extent, xl, yl = slices[i]
        imgf = img.astype(float)
        imgf[imgf < 1] = np.nan
        n_classes = max(r.class_means)
        colors = zone_colors(r.class_means)
        cmap = ListedColormap([colors.get(c, (0.5, 0.5, 0.5, 1.0))
                               for c in range(1, n_classes + 1)])
        im = ax.imshow(imgf, origin="lower", extent=extent, cmap=cmap,
                       vmin=0.5, vmax=n_classes + 0.5, interpolation="nearest")
        unit = var_unit(r.var_name)
        cbar = fig.colorbar(im, ax=ax, ticks=range(1, n_classes + 1),
                            fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels([zone_label(c, r.class_means, unit)
                                 for c in range(1, n_classes + 1)], fontsize=7)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_title(r.case_name, fontsize=11)
    fig.suptitle(f"Zone maps by case \u2014 {results[0].var_name}", fontsize=13)
    fig.tight_layout()
    fig_to_html(fig, f"Zone maps \u2014 {results[0].var_name}", out_path)


def batch_html_2d_common(results: list[CaseResult], out_path: Path) -> None:
    """Zone maps on one shared color scale: zones with the same mean get the same
    color across cases, so differences between cases are directly visible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    var_name = results[0].var_name
    unit = var_unit(var_name)
    # shared discrete scale over the unique zone-mean values of all cases
    key = lambda m: float(f"{m:.3g}")
    uniq = sorted({key(m) for r in results for m in r.class_means.values()})
    val_colors = zone_colors(dict(enumerate(uniq)))
    color_of = {v: val_colors[i] for i, v in enumerate(uniq)}

    n = len(results)
    ncols, nrows = _case_grid(n)
    slices = [zone_slice_2d(r) for r in results]
    limits, box_aspect = _grouped_limits(slices)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols + 2.0, 5.5 * nrows),
                             squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for i, r in enumerate(results):
        ax = axes.ravel()[i]
        img, extent, xl, yl = slices[i]
        imgf = img.astype(float)
        imgf[imgf < 1] = np.nan
        n_classes = max(r.class_means)
        cmap = ListedColormap([color_of.get(key(r.class_means[c]), (0.5, 0.5, 0.5, 1.0))
                               if c in r.class_means else (0.5, 0.5, 0.5, 1.0)
                               for c in range(1, n_classes + 1)])
        ax.imshow(imgf, origin="lower", extent=extent, cmap=cmap,
                  vmin=0.5, vmax=n_classes + 0.5, interpolation="nearest",
                  aspect="auto")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        x0, x1, y0, y1 = limits[i]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_box_aspect(box_aspect)
        ax.set_title(r.case_name, fontsize=11)
    # one shared colorbar over the pooled zone-mean values
    sm = plt.cm.ScalarMappable(
        cmap=ListedColormap([color_of[v] for v in uniq]),
        norm=BoundaryNorm(np.arange(len(uniq) + 1), len(uniq)))
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.05, pad=0.02,
                        ticks=np.arange(len(uniq)) + 0.5)
    cbar.ax.set_yticklabels([f"{v:.3g}{unit}" for v in uniq], fontsize=8)
    cbar.set_label(f"zone mean {var_name}")
    fig.suptitle(f"Zone maps on a common color scale \u2014 {var_name}", fontsize=13)
    fig_to_html(fig, f"Zone maps (common scale) \u2014 {var_name}", out_path)


def batch_html_2d_normalized(results: list[CaseResult], out_path: Path) -> None:
    """Zone maps colored by zone mean on a continuous scale normalized to the
    lowest/highest zone mean actually visible in the displayed slices."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    var_name = results[0].var_name
    n = len(results)
    ncols, nrows = _case_grid(n)
    slices = [zone_slice_2d(r) for r in results]
    # color scale spans only the zone means of classes present in the slices,
    # so unseen zones (e.g. tiny near-zero pockets) don't stretch the scale
    vis_means = [r.class_means[int(c)]
                 for r, (img, _, _, _) in zip(results, slices)
                 for c in np.unique(img[img >= 1]) if int(c) in r.class_means]
    if not vis_means:
        vis_means = [m for r in results for m in r.class_means.values()]
    pos_means = [m for m in vis_means if m > 0]
    vmin = min(pos_means) if pos_means else min(vis_means)
    vmax = max(vis_means)
    norm = LogNorm(vmin, vmax) if vmin > 0 else Normalize(vmin, vmax)

    limits, box_aspect = _grouped_limits(slices)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols + 1.5, 5.5 * nrows),
                             squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    im = None
    for i, r in enumerate(results):
        ax = axes.ravel()[i]
        img, extent, xl, yl = slices[i]
        lut = np.full(max(r.class_means) + 1, np.nan)
        for c, m in r.class_means.items():
            lut[c] = m
        vals = np.where(img >= 1, lut[np.clip(img, 0, len(lut) - 1)], np.nan)
        im = ax.imshow(vals, origin="lower", extent=extent, cmap="turbo", norm=norm,
                       interpolation="nearest", aspect="auto")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        x0, x1, y0, y1 = limits[i]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_box_aspect(box_aspect)
        ax.set_title(r.case_name, fontsize=11)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.04, pad=0.02)
    cbar.set_label(f"zone mean {var_name}")
    fig.suptitle(f"Zone means on a continuous scale, displayed-zone range "
                 f"({vmin:.3g} .. {vmax:.3g}) \u2014 {var_name}", fontsize=13)
    fig_to_html(fig, f"Zone means (normalized) \u2014 {var_name}", out_path)


def batch_exposure_cdf(results: list[CaseResult], out_path: Path) -> None:
    """Combined volume-exposure curves of all cases (values reloaded from zones.vti)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    any_pos = False
    for r in results:
        grid, zone_class, _ = load_zone_class(r)
        var = np.asarray(grid.cell_data[r.var_name]).reshape(zone_class.shape)
        v = np.sort(var[zone_class > 0].astype(np.float64))
        if (v > 0).any():
            any_pos = True
        idx = np.linspace(0, v.size - 1, min(4000, v.size)).astype(int)
        line, = ax.plot(v[idx], 100 * (idx + 1) / v.size, lw=1.5, label=r.case_name)
        ax.axvline(r.global_mean, color=line.get_color(), ls="--", lw=1)
    if any_pos:
        ax.set_xscale("log")
    ax.set_xlabel(results[0].var_name)
    ax.set_ylabel("cumulative fluid volume [%]")
    ax.set_title("Volume exposure by case (dashed: case global means)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"wrote {out_path}")


def batch_html_3d(results: list[CaseResult], out_path: Path) -> None:
    """Single interactive HTML with linked 3D zone views, one subplot per case."""
    import pyvista as pv

    n = len(results)
    ncols = min(3, n)
    nrows = -(-n // ncols)
    pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                    window_size=(640 * ncols, 540 * nrows))
    for i, r in enumerate(results):
        pl.subplot(i // ncols, i % ncols)
        grid, _, _ = load_zone_class(r)
        colors = zone_colors(r.class_means)
        unit = var_unit(r.var_name)
        for c in sorted(r.class_means):
            zone = grid.threshold([c - 0.5, c + 0.5], scalars="ZoneClass")
            if zone.n_cells:
                zone = zone.clip(normal="z", origin=grid.center, invert=False)  # cutaway
            if zone.n_cells == 0:  # zone may vanish entirely in the cutaway
                continue
            pl.add_mesh(zone, color=colors[c][:3], opacity=0.95 if c == 1 else 0.6,
                        label=zone_label(c, r.class_means, unit))
        if r.model.impeller_stl is not None:
            try:
                pl.add_mesh(pv.read(str(r.model.impeller_stl)), color="dimgray")
            except Exception:
                pass
        pl.add_text(r.case_name, font_size=10)
        # 2D overlays are dropped by the HTML export; 3D text above the tank survives
        try:
            b = grid.bounds
            up_i = (int(np.argmax(np.abs(r.model.impeller_axis)))
                    if r.model.impeller_axis is not None else 1)
            h = 0.06 * max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
            center = list(grid.center)
            center[up_i] = b[2 * up_i + 1] + 1.5 * h
            if up_i != 2:
                center[2] = b[5]  # float in front of the tank, clear of the shaft
            txt = pv.Text3D(r.case_name, height=h, depth=0.1 * h,
                            center=center, normal=(0.0, 0.0, 1.0))
            pl.add_mesh(txt, color="black")
        except Exception:
            pass
        up = (tuple(np.abs(r.model.impeller_axis))
              if r.model.impeller_axis is not None else (0, 1, 0))
        pl.view_vector((1.0, 0.6, 1.0), viewup=up)
        pl.reset_camera(bounds=grid.bounds)
    try:
        pl.link_views()
    except Exception:
        pass
    try:
        pl.export_html(str(out_path))
        log(f"wrote {out_path}")
    except Exception as e:  # headless rendering can fail without a display/GPU
        warn(f"3D comparison HTML failed ({e})")


def case_out_dir(base: Path, variables: list[str], variable: str) -> Path:
    """Flat output dir for a single variable, per-variable subdirs otherwise."""
    return base if len(variables) == 1 else base / var_slug(variable)


def load_case_result(case: Case, model: ModelInfo, out_dir: Path) -> CaseResult | None:
    """Rebuild a CaseResult from a previous run's zoner_output, for --comparison-only."""
    zones_csv, zones_vti = out_dir / "zones.csv", out_dir / "zones.vti"
    if not zones_csv.is_file() or not zones_vti.is_file():
        return None

    class_rows: list[tuple[int, str, dict]] = []
    with open(zones_csv, newline="") as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        var_name = header[6].removeprefix("mean ")
        for row in rdr:
            if row[0] != "class":
                continue
            s = {"cells": int(row[3]), "volume_m3": float(row[4]),
                 "vol_frac": float(row[5]), "mean": float(row[6]),
                 "std": float(row[7]), "min": float(row[8]), "max": float(row[9])}
            class_rows.append((int(row[1]), row[2], s))
    if not class_rows:
        return None
    class_means = {c: s["mean"] for c, _, s in class_rows}

    het: dict = {}
    het_csv = out_dir / "heterogeneity.csv"
    if het_csv.is_file():
        with open(het_csv, newline="") as fh:
            rdr = csv.reader(fh)
            next(rdr)
            for k, v in rdr:
                het[k] = v == "True" if k == "significant_gradients" else float(v)

    import pyvista as pv

    grid = pv.read(str(zones_vti))
    zone_class = np.asarray(grid.cell_data["ZoneClass"])
    var = np.asarray(grid.cell_data[var_name], dtype=np.float64)
    global_mean = float(var[zone_class > 0].mean())

    log(f"reloaded existing results from {out_dir}")
    return CaseResult(case_name=case.root.name, var_name=var_name, t_sel=float("nan"),
                      global_mean=global_mean, class_rows=class_rows,
                      class_means=class_means, out_dir=out_dir, model=model, het=het)


def run_batch(args) -> None:
    root = args.case
    if not root.is_dir():
        fail(f"batch root not found: {root}")
    cases = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith(".") and d.name not in SKIP_DIRS:
            c = discover_case(d)
            if c.vti_series:
                cases.append(c)
    if not cases:
        fail(f"no case subdirectories with VTI output found under {root}")
    log(f"batch mode: {len(cases)} case(s): {', '.join(c.root.name for c in cases)}")
    if args.n_zones is None:
        warn("no --n-zones given: auto-k may differ per case, weakening rank alignment")

    if args.list_vars:
        for n in list_vti_arrays(cases[0].vti_series[0][1]):
            print(n)
        return

    results: dict[str, list[CaseResult]] = {v: [] for v in args.variable}
    for case in cases:
        log(f"--- case: {case.root.name} ---")
        log(f"found {len(case.vti_series)} VTI timesteps: "
            f"t = {case.vti_series[0][0]:.3g} .. {case.vti_series[-1][0]:.3g} s")
        model = parse_input_xml(case.input_xml, case.stl_files)
        base = (args.output_dir / case.root.name) if args.output_dir else case.root / "zoner_output"
        for v in args.variable:
            out_dir = case_out_dir(base, args.variable, v)
            try:
                if args.comparison_only:
                    # earlier runs may have used the other layout (flat vs per-variable)
                    alt_dir = base / var_slug(v) if out_dir == base else base
                    r = (load_case_result(case, model, out_dir)
                         or load_case_result(case, model, alt_dir))
                    if r is None:
                        warn(f"case '{case.root.name}', variable '{v}': no existing "
                             f"results in {out_dir} or {alt_dir}; skipping (rerun "
                             f"without --comparison-only to generate them)")
                    else:
                        results[v].append(r)
                else:
                    results[v].append(run_case(case, model, v, args, out_dir))
            except (Exception, SystemExit) as e:  # keep the batch alive on bad cases
                warn(f"case '{case.root.name}', variable '{v}' failed: {e}")

    comp_base = (args.output_dir / "zoner_comparison") if args.output_dir else root / "zoner_comparison"
    for v in args.variable:
        if len(results[v]) < 2:
            warn(f"variable '{v}': only {len(results[v])} case(s) succeeded; skipping comparison")
            continue
        comp_dir = case_out_dir(comp_base, args.variable, v)
        compare_cases(results[v], comp_dir)
        batch_html_2d(results[v], comp_dir / "zones_2d.html")
        batch_html_2d_common(results[v], comp_dir / "zones_2d_common.html")
        batch_html_2d_normalized(results[v], comp_dir / "zones_2d_normalized.html")
        batch_exposure_cdf(results[v], comp_dir / "exposure_cdf.png")
        batch_html_3d(results[v], comp_dir / "zones_3d.html")


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("case", type=Path,
                    help="M-Star case directory or single .vti file; with --batch, "
                         "a root directory containing case subdirectories")
    ap.add_argument("--batch", action="store_true",
                    help="process every case subdirectory under 'case' and compare "
                         "zones across cases")
    ap.add_argument("--comparison-only", action="store_true",
                    help="with --batch: reuse existing per-case zoner_output data "
                         "and regenerate only the cross-case comparison")
    ap.add_argument("--variable", nargs="+", default=["edr"],
                    help="variable(s) to analyze (fuzzy match, e.g. 'edr', 'shear', 'tke')")
    ap.add_argument("--n-zones", type=int, default=None,
                    help="total zones incl. impeller; omit for automatic selection")
    ap.add_argument("--time", type=float, default=None,
                    help="analysis time [s]; overrides steady-state detection")
    ap.add_argument("--steady-on", choices=["variable", "power"], default="variable",
                    help="steady-state criterion source")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="steady-state tolerance on windowed-mean drift")
    ap.add_argument("--window", type=float, default=1.0, help="steady-state window [s]")
    ap.add_argument("--instantaneous", action="store_true",
                    help="use instantaneous field even if a Time-Avg variant exists")
    ap.add_argument("--linear", action="store_true",
                    help="cluster on linear values, not log-transformed")
    ap.add_argument("--log-base", type=float, default=10.0,
                    help="log base (> 1) for zoning; with the default k-means "
                         "zoning it is cosmetic, with --log-bins it sets the "
                         "bin width (ignored with --linear)")
    ap.add_argument("--log-bins", action="store_true",
                    help="zone by fixed 1-log-unit bins instead of k-means: "
                         "class boundaries at powers of --log-base, e.g. orders "
                         "of magnitude (base 10) or doublings (base 2); zone "
                         "count follows the data range")
    ap.add_argument("--min-zone-frac", type=float, default=0.005,
                    help="min sub-zone size as fraction of fluid volume")
    ap.add_argument("--eta2-threshold", type=float, default=0.5,
                    help="between-zone variance fraction above which gradients "
                         "are flagged significant")
    ap.add_argument("--contrast-threshold", type=float, default=3.0,
                    help="zone contrast (max/min zone mean) above which gradients "
                         "are flagged significant")
    ap.add_argument("--impeller-pad", type=float, default=0.15,
                    help="impeller cylinder padding fraction")
    ap.add_argument("--impeller-percentile", type=float, default=99.0,
                    help="velocity percentile for fallback impeller detection")
    ap.add_argument("--impeller-shape", choices=["cylinder", "blob"], default="cylinder",
                    help="fallback impeller zone shape")
    ap.add_argument("--no-impeller-zone", action="store_true",
                    help="skip the geometric impeller zone; all n_zones come from clustering")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="output directory (default: <case>/zoner_output; in batch mode, "
                         "per-case subdirs are created under it)")
    ap.add_argument("--list-vars", action="store_true", help="list VTI arrays and exit")
    ap.add_argument("--no-plots", action="store_true", help="skip diagnostic plots")
    ap.add_argument("--no-render", action="store_true", help="skip 3D rendering")
    ap.add_argument("--show", action="store_true", help="open interactive 3D window")
    ap.add_argument("--html", action="store_true",
                    help="export interactive 3D HTML and a .glb 3D model (for PowerPoint)")
    args = ap.parse_args(argv)

    global HAS_IMPELLER
    HAS_IMPELLER = not args.no_impeller_zone
    if args.n_zones is not None and args.n_zones < 2:
        fail("--n-zones must be >= 2")
    if args.log_base <= 1:
        fail("--log-base must be > 1")
    if args.comparison_only and not args.batch:
        fail("--comparison-only requires --batch")

    if args.batch:
        run_batch(args)
        log("done")
        return

    case = discover_case(args.case)
    if not case.vti_series:
        fail(f"no .vti files found under {args.case}")
    log(f"found {len(case.vti_series)} VTI timesteps: "
        f"t = {case.vti_series[0][0]:.3g} .. {case.vti_series[-1][0]:.3g} s")

    if args.list_vars:
        for n in list_vti_arrays(case.vti_series[0][1]):
            print(n)
        return

    model = parse_input_xml(case.input_xml, case.stl_files)
    base = args.output_dir or (args.case if args.case.is_dir() else args.case.parent) / "zoner_output"
    for v in args.variable:
        if len(args.variable) > 1:
            log(f"--- variable: {v} ---")
        run_case(case, model, v, args, case_out_dir(base, args.variable, v))
    log("done")


if __name__ == "__main__":
    main()
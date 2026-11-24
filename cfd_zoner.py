#!/usr/bin/env python3
"""Zonal analysis of M-Star CFD volume (VTI) output.

Reads an M-Star results directory in its native layout (case/input.xml,
case/*.stl, case/out/Stats/, case/out/Output/Volume/block*/*.vti; the blocks of a
multi-GPU solve are merged into one grid), finds momentum steady state (or uses a
user-defined time), analyses the Time-Avg array of the last frame or else the average
of the instantaneous field over the frames from the steady start on, always defines an
impeller zone, clusters the selected variable into value-based zones (log-scale
k-means) split into spatially contiguous sub-zones, and reports per-zone stats,
a labeled VTI for ParaView, diagnostic plots, and a 3D rendering.

Example:
    python cfd_zoner.py test_case_1 --variable edr --n-zones 4
    python cfd_zoner.py sim_root --batch --variable edr shear --n-zones 4
"""

from __future__ import annotations

import os

# before numpy is imported: BLAS/OpenMP thread pools oversubscribe a many-core host
for _threads in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_threads, "4")

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
    """Normalize separators and strip unit suffixes like '(W/kg)' or '[W/kg]'."""
    s = re.sub(r"[\[\(][^\]\)]*[\]\)]", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


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
    stats_scalars: list[Path] = field(default_factory=list)
    stl_files: list[Path] = field(default_factory=list)


# M-Star encodes sim time in the filename, e.g. Volume.5.00000e+00.vti
TIME_RE = re.compile(r"\.([0-9]+(?:\.[0-9]+)?(?:e[+-][0-9]+)?)$", re.IGNORECASE)

# directories the tool writes itself (never treated as simulation input)
SKIP_DIRS = {"zoner_output", "zoner_comparison"}


def _rglob(root: Path, pattern: str) -> list[Path]:
    return sorted(f for f in root.rglob(pattern)
                  if not SKIP_DIRS.intersection(f.parts))


# multi-GPU solves write one piece per GPU: block0, block1, ... (same file names)
BLOCK_RE = re.compile(r"^block(\d+)$", re.IGNORECASE)


def block_dirs(parent: Path) -> list[Path]:
    dirs = [d for d in parent.iterdir() if d.is_dir() and BLOCK_RE.match(d.name)]
    return sorted(dirs, key=lambda d: int(BLOCK_RE.match(d.name).group(1)))


def block_files(path: Path) -> list[Path]:
    """The same-named file of every blockN folder next to path's own block folder;
    [path] for single-block output. A frame missing in one block is an error."""
    if not BLOCK_RE.match(path.parent.name):
        return [path]
    files = [d / path.name for d in block_dirs(path.parent.parent)]
    missing = [f.parent.name for f in files if not f.is_file()]
    if missing:
        fail(f"frame {path.name} is incomplete: no such file in {', '.join(missing)} "
             f"(found in {len(files) - len(missing)} of {len(files)} blocks)")
    return files


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
    pvd_times: dict[Path, float] | None = None  # parsed only for names without a time

    series = []
    unparsed = []
    for f in vti_files:
        m = TIME_RE.search(f.stem)
        if m:
            series.append((float(m.group(1)), f))
            continue
        if pvd_times is None:
            pvd_times = {}
            for pvd in pvd_files:
                pvd_times.update(read_pvd_times(pvd))
        if f.resolve() in pvd_times:
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
        blocks = block_dirs(volume_dir)
        if len(blocks) > 1:
            log(f"{len(blocks)} volume blocks ({', '.join(b.name for b in blocks)}): "
                f"merged into one grid per frame")
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
    case.stats_scalars = sorted(stats_root.glob("Scalar_*.txt")) or _rglob(root, "Scalar_*.txt")

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
    impeller_name: str | None = None
    impeller_scale: float = 1.0
    impeller_matrix: np.ndarray | None = None  # <transform> 4x4, row vectors: p' = [p, 1] @ M
    impeller_pivot: np.ndarray | None = None   # a point on the rotation axis


def find_stl(name: str, stl_files: list[Path]) -> Path | None:
    for f in stl_files:
        if f.name.lower() == name.lower():
            return f
    return None


def _floats(text: str | None, n: int) -> np.ndarray | None:
    try:
        v = np.array([float(x) for x in (text or "").split()])
    except ValueError:
        return None
    return v if v.size == n else None


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
            info.impeller_name = (imp.findtext("name") or "").strip() or None
            scale = _floats(imp.findtext("scale"), 1)
            info.impeller_scale = float(scale[0]) if scale is not None else 1.0
            matrix = _floats(imp.findtext("transform/matrix4x4"), 16)
            info.impeller_matrix = matrix.reshape(4, 4) if matrix is not None else None
            info.impeller_pivot = _floats(imp.findtext("pivot"), 3)
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
        if info.impeller_stl is not None and impeller_is_placed(info):
            m = info.impeller_matrix if info.impeller_matrix is not None else np.eye(4)
            log(f"impeller STL placed as in input.xml: scale {info.impeller_scale:g}, "
                f"translation {m[3, :3]}"
                + ("" if np.allclose(m[:3, :3], np.eye(3)) else ", with rotation"))
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


def impeller_is_placed(model: ModelInfo) -> bool:
    """True when input.xml scales or transforms the impeller STL."""
    m = model.impeller_matrix
    return model.impeller_scale != 1.0 or (m is not None and not np.array_equal(m, np.eye(4)))


def load_impeller(model: ModelInfo):
    """The impeller STL where M-Star puts it (initial pose): scaled, then transformed
    with the input.xml matrix (row vectors, translation in the last row)."""
    import pyvista as pv

    mesh = pv.read(str(model.impeller_stl))
    if impeller_is_placed(model):
        pts = np.asarray(mesh.points, dtype=np.float64) * model.impeller_scale
        if model.impeller_matrix is not None:
            pts = pts @ model.impeller_matrix[:3, :3] + model.impeller_matrix[3, :3]
        mesh.points = pts
    return mesh


def up_axis(axis: np.ndarray | None) -> np.ndarray:
    """The shaft direction, pointing to the positive side of its dominant coordinate."""
    if axis is None:
        return np.array([0.0, 1.0, 0.0])
    return -axis if axis[int(np.argmax(np.abs(axis)))] < 0 else axis + 0.0


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
        if target in h and "time avg" not in h:
            return i
    return None


def find_scalar_stats(case: Case, var_query: str) -> Path | None:
    """Find the standalone M-Star Scalar_<object>.txt matching a field query."""
    query = norm_name(var_query)
    query_core = re.sub(r"\b(concentration|conc)\b", "", query).strip()
    matches = []
    for path in case.stats_scalars:
        stem = norm_name(path.stem)
        if query == stem or query_core == stem or query in stem or stem in query_core:
            matches.append(path)
    if len(matches) > 1:
        warn(f"multiple scalar stats files match '{var_query}'; using {matches[0].name}")
    return matches[0] if matches else None


def scalar_stats_at_time(case: Case, var_query: str, target_time: float) -> dict | None:
    """Return M-Star's scalar mean and RSD nearest the selected VTI time."""
    path = find_scalar_stats(case, var_query)
    if path is None:
        return None
    header, data = load_stats_table(path)
    mean_col = find_column(header, "conc mean")
    if mean_col is None:
        mean_col = find_column(header, "mean")
    rsd_col = find_column(header, "conc rsd")
    if rsd_col is None:
        rsd_col = find_column(header, "rsd")
    if mean_col is None:
        warn(f"no mean column found in {path.name}")
        return None
    idx = int(np.argmin(np.abs(data[:, 0] - target_time)))
    result = {"time_s": float(data[idx, 0]), "mean": float(data[idx, mean_col]),
              "source": path.name}
    if rsd_col is not None:
        result["rsd_percent"] = float(data[idx, rsd_col])
    return result


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
        src = find_scalar_stats(case, var_query)
        if src is not None:
            header, data = load_stats_table(src)
            col = find_column(header, "conc mean")
            if col is None:
                col = find_column(header, "mean")
        elif case.stats_fluid is not None:
            header, data = load_stats_table(case.stats_fluid)
            col = find_column(header, f"mean {qn}")
            if col is None:
                col = find_column(header, qn)
            if col is None and qn.endswith(" magnitude"):  # Fluid.txt has 'Mean Velocity [m/s]'
                col = find_column(header, f"mean {qn[:-len(' magnitude')]}")
            src = case.stats_fluid
        else:
            return None
    if col is None:
        warn(f"no matching column in {src.name}")
        return None
    log(f"steady-state trace: '{header[col]}' from {src.name}")
    return data[:, 0], data[:, col], header[col]


EXCURSION_SHARE = 0.10  # unsteady samples tolerated inside a plateau of >= window / share


def plateau_index(times: np.ndarray, steady: np.ndarray, window: float) -> int | None:
    """Index of the first sample of the plateau, given the per-sample steady flags
    (same rule as plateau_index in mstar_stats_summary.py).

    Strictly, the plateau starts at the first steady sample after which every sample is
    steady. Short excursions are tolerated in a plateau long enough to judge them: runs
    of unsteady samples lasting less than one window, making up at most EXCURSION_SHARE
    of the samples of a plateau spanning at least window / EXCURSION_SHARE. A run of a
    window or longer is a real change and the plateau starts after it. None if there is
    no plateau."""
    n = len(times)
    if n == 0:
        return None
    bad = ~steady
    edges = np.flatnonzero(np.diff(np.concatenate(([0], bad.astype(np.int8), [0]))))
    first, last = edges[::2], edges[1::2] - 1  # inclusive ends of each unsteady run
    real = times[last] - times[first] >= window
    i0 = int(last[real][-1]) + 1 if real.any() else 0
    share = np.cumsum(bad[::-1])[::-1] / (n - np.arange(n))
    ok = steady & (share <= EXCURSION_SHARE) & (times[-1] - times >= window / EXCURSION_SHARE)
    ok[:i0] = False
    ok |= steady & (share == 0)
    hit = np.flatnonzero(ok)
    return int(hit[0]) if hit.size else None


def steady_flags(times: np.ndarray, values: np.ndarray, window: float, tol: float) -> np.ndarray:
    """flags[i]: the mean of the window ending at sample i differs from the mean of the
    window before it by less than tol (False until two full windows have passed)."""
    n = len(times)
    flags = np.zeros(n, bool)

    def one(i: int) -> bool:  # the definition, sample by sample
        t = times[i]
        cur = (times > t - window) & (times <= t)
        prev = (times > t - 2 * window) & (times <= t - window)
        if t < times[0] + 2 * window or cur.sum() < 3 or prev.sum() < 3:
            return False
        m_cur, m_prev = values[cur].mean(), values[prev].mean()
        return bool(abs(m_prev) > 0 and abs(m_cur - m_prev) / abs(m_prev) < tol)

    if n < 2 or not np.all(np.diff(times) >= 0):  # unsorted times: no window slices
        todo = np.arange(n)
    else:  # window means from one running sum; samples the rounding could flip are redone
        hi = np.searchsorted(times, times, side="right")
        mid = np.searchsorted(times, times - window, side="right")
        lo = np.searchsorted(times, times - 2 * window, side="right")
        n_cur, n_prev = hi - mid, mid - lo
        usable = ~(times < times[0] + 2 * window) & (n_cur >= 3) & (n_prev >= 3)
        total = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
        err = 4 * n * np.finfo(np.float64).eps * float(np.abs(values).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            m_cur = (total[hi] - total[mid]) / n_cur
            m_prev = (total[mid] - total[lo]) / n_prev
            change = np.abs(m_cur - m_prev) / np.abs(m_prev)
            slack = err / np.minimum(n_cur, n_prev) * (2 + change) / np.abs(m_prev)
            sure = np.abs(change - tol) > slack
        flags = usable & sure & (change < tol)
        todo = np.flatnonzero(usable & ~sure)
    for i in todo:
        flags[i] = one(i)
    return flags


def detect_steady_state(times: np.ndarray, values: np.ndarray, window: float,
                        tol: float) -> float | None:
    """Start of the steady plateau of the windowed mean (change < tol), or None.

    Compares the mean of the trailing window against the window before it, which
    tolerates turbulent fluctuations that never settle below tol themselves; short
    excursions inside a long plateau are tolerated too (plateau_index).
    """
    flags = steady_flags(times, values, window, tol)
    idx = plateau_index(times, flags, window)
    return None if idx is None else float(times[idx])


# ---------------------------------------------------------------- VTI handling

def vti_header_text(path: Path) -> str:
    """The XML header of a .vti file (everything before the binary payload)."""
    head = b""
    with open(path, "rb") as fh:
        while b"<AppendedData" not in head and len(head) < 4_000_000:
            chunk = fh.read(65536)
            if not chunk:
                break
            head += chunk
    return head.decode("utf-8", errors="ignore")


def list_vti_arrays(path: Path) -> list[str]:
    """Read array names from the XML header without loading the binary payload."""
    return re.findall(r'DataArray[^>]*Name="([^"]+)"', vti_header_text(path))


def scalar_cell_arrays(path: Path) -> list[str]:
    """Names of the one-component cell arrays in file order, from the XML header."""
    import html

    sec = re.search(r"<CellData[^>]*>(.*?)</CellData>", vti_header_text(path), re.S)
    names = []
    for tag in re.findall(r"<DataArray[^>]*>", sec.group(1)) if sec else []:
        name = re.search(r'Name="([^"]+)"', tag)
        ncomp = re.search(r'NumberOfComponents="(\d+)"', tag)
        if name and (ncomp is None or int(ncomp.group(1)) == 1):
            names.append(html.unescape(name.group(1)))
    if names:
        return names
    import pyvista as pv  # header not readable this way: load the file once

    grid = pv.read(str(path))
    return [n for n in grid.cell_data.keys() if grid.cell_data[n].ndim == 1]


def resolve_variable(query: str, names: list[str], prefer_time_avg: bool) -> str:
    qn = VAR_ALIASES.get(query.lower().strip(), norm_name(query))
    normed = {n: norm_name(n) for n in names}
    want_avg = "time avg" in qn  # norm_name turns 'Time-Avg' into 'time avg'

    def match(pool):
        exact = [n for n in pool if normed[n] == qn]
        if exact:
            return exact[0]
        sub = [n for n in pool if qn in normed[n]]
        return sub[0] if sub else None

    base_pool = names if want_avg else [n for n in names if "time avg" not in normed[n]]
    hit = match(base_pool) or match(names)
    if hit is None:
        fail(f"variable '{query}' not found. Available: {names}")
    if prefer_time_avg and not want_avg:
        avg = [n for n in names if normed[n] == f"time avg {normed[hit]}"]
        if avg:
            hit = avg[0]
    return hit


@dataclass
class Field3D:
    grid: "object"                 # pyvista ImageData (geometry only)
    nc: tuple[int, int, int]       # cells (nx, ny, nz)
    origin: np.ndarray
    spacing: np.ndarray
    var_name: str
    var: np.ndarray                # [z, y, x]
    velmag: np.ndarray | None      # [z, y, x]
    covered: np.ndarray | None = None  # [z, y, x] cells present in a block file; None = all
    solid: np.ndarray | None = None    # [z, y, x] cells inside the moving body
    body: str = ""                     # where `solid` comes from, for the log

    def cell_centers_1d(self):
        o, sp, nc = self.origin, self.spacing, self.nc
        return tuple(o[i] + (np.arange(nc[i], dtype=np.float32) + 0.5) * sp[i] for i in range(3))

    def cell_centers_3d(self):
        xc, yc, zc = self.cell_centers_1d()
        Z, Y, X = np.meshgrid(zc, yc, xc, indexing="ij")
        return X, Y, Z

    def coordinate(self, i: int, mask: np.ndarray) -> np.ndarray:
        """Coordinate i (0 = x, 1 = y, 2 = z) of the cell centres selected by mask."""
        shape = [1, 1, 1]
        shape[2 - i] = -1
        return np.broadcast_to(self.cell_centers_1d()[i].reshape(shape), mask.shape)[mask]


def read_frame(path: Path, names: list[str]) -> tuple:
    """Selected cell arrays of one frame as float32 [z, y, x]: (nc, origin, spacing,
    {name: array}, covered). Multi-GPU output (block0, block1, ...) is assembled into one
    grid by each block's origin and extent; cells no block holds stay 0 and are flagged
    in `covered` (None when every cell is covered)."""
    import pyvista as pv

    parts = []
    for f in block_files(path):
        reader = pv.get_reader(str(f))
        reader.disable_all_cell_arrays()
        reader.disable_all_point_arrays()
        for n in names:
            reader.enable_cell_array(n)
        g = reader.read()
        absent = [n for n in names if n not in g.cell_data]
        if absent:
            fail(f"{f.parent.name}/{f.name} has no cell array {absent}")
        nc = tuple(int(d) - 1 for d in g.dimensions)
        sp = np.array(g.spacing)
        lo = np.array(g.origin) + np.array(g.extent[::2]) * sp
        parts.append((nc, lo, sp, [np.asarray(g.cell_data[n], dtype=np.float32)
                                   .reshape((nc[2], nc[1], nc[0])) for n in names]))
    nc, lo, sp, arrays = parts[0]
    if len(parts) == 1:
        return nc, lo, sp, dict(zip(names, arrays)), None
    if any(not np.allclose(p[2], sp, rtol=1e-6, atol=0) for p in parts):
        fail(f"the blocks of {path.name} differ in grid spacing")
    lo = np.min([p[1] for p in parts], axis=0)
    offs = [np.rint((p[1] - lo) / sp).astype(int) for p in parts]
    nc = tuple(int(max(o[a] + p[0][a] for o, p in zip(offs, parts))) for a in range(3))
    shape = (nc[2], nc[1], nc[0])
    out = {n: np.zeros(shape, np.float32) for n in names}
    covered = np.zeros(shape, bool)
    for (bnc, _, _, arrays), o in zip(parts, offs):
        sl = tuple(slice(int(o[a]), int(o[a]) + bnc[a]) for a in (2, 1, 0))
        covered[sl] = True
        for n, a in zip(names, arrays):
            out[n][sl] = a
    return nc, lo, sp, out, None if covered.all() else covered


class MovingBody:
    """The cells inside the impeller. Moving-body cells carry the solid's velocity, so the
    zero-velocity blanking misses them. They are taken from M-Star's posed mesh of the
    frame (out/Output/VolumeMovingBody_<name>/block*/*.<time>.vtp) when the case has
    it, else from the STL in its initial pose, placed as in input.xml."""

    def __init__(self, case: Case, model: ModelInfo):
        self.model = model
        self.dirs: list[Path] = []
        self.series: list[tuple[float, str]] = []
        self.static: np.ndarray | None = None  # mask of the STL in its initial pose
        self.stl = load_impeller(model) if model.impeller_stl is not None else None
        self.name = model.impeller_stl.name if model.impeller_stl is not None else ""
        if model.impeller_name and case.root.is_dir():
            d = case.root / "out" / "Output" / f"VolumeMovingBody_{model.impeller_name}"
            if d.is_dir():
                self.dirs = block_dirs(d) or [d]
                found = ((TIME_RE.search(f.stem), f.name) for f in self.dirs[0].glob("*.vtp"))
                self.series = sorted((float(m.group(1)), name) for m, name in found if m)
                self.name = d.name
        self.posed = bool(self.series)

    def _posed_surface(self, t: float):
        import pyvista as pv

        dt, t_mesh, name = min((abs(tm - t), tm, name) for tm, name in self.series)
        if dt > 1e-5 * abs(t) + 1e-9:
            warn(f"no posed moving-body mesh at t = {t:.6g} s; using the nearest one "
                 f"(t = {t_mesh:.6g} s)")
        parts = [pv.read(str(d / name)) for d in self.dirs if (d / name).is_file()]
        parts = [p for p in parts if p.n_cells]
        if not parts:
            raise ValueError(f"{name} holds no surface")
        surf = parts[0] if len(parts) == 1 else parts[0].append_polydata(*parts[1:])
        return surf if surf.is_all_triangles else surf.triangulate()

    def _turned_stl(self, posed: np.ndarray):
        """The STL turned about the impeller axis into the pose of a posed mesh, or None
        when no such rotation puts every STL vertex on a vertex of that mesh. The posed
        mesh is M-Star's refinement of the same surface with ~50x the triangles, so the
        STL gives the same cells in a fraction of the time."""
        from scipy.spatial import cKDTree

        axis, pivot = self.model.impeller_axis, self.model.impeller_pivot
        if self.stl is None or axis is None or pivot is None:
            return None
        e1 = np.cross(axis, np.eye(3)[int(np.argmin(np.abs(axis)))])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)

        def cylindrical(points):
            p = points - pivot
            u, v = p @ e1, p @ e2
            return p, p @ axis, np.hypot(u, v), np.arctan2(v, u)

        s, s_ax, s_rad, s_phi = cylindrical(np.asarray(self.stl.points, dtype=np.float64))
        _, p_ax, p_rad, p_phi = cylindrical(posed)
        tol = 1e-5 * float(np.linalg.norm(s.max(axis=0) - s.min(axis=0)))
        k = int(np.argmax(s_rad))  # the outermost STL vertex: where is it in the posed mesh?
        tree = cKDTree(posed)
        for j in np.flatnonzero((np.abs(p_rad - s_rad[k]) < tol) & (np.abs(p_ax - s_ax[k]) < tol))[:32]:
            c, sn = np.cos(p_phi[j] - s_phi[k]), np.sin(p_phi[j] - s_phi[k])
            turned = s * c + np.cross(axis, s) * sn + np.outer(s @ axis, axis) * (1 - c) + pivot
            if tree.query(turned)[0].max() < tol:
                mesh = self.stl.copy()
                mesh.points = turned
                return mesh
        return None

    def mask(self, fld: Field3D, t: float) -> np.ndarray | None:
        """Cells inside the body at time t (the STL mask when there is no posed mesh)."""
        if self.static is None and self.stl is not None:
            self.static = stl_inside_mask(fld, self.stl)
        if not self.posed:
            return self.static
        try:
            surf = self._posed_surface(t)
            turned = self._turned_stl(np.asarray(surf.points, dtype=np.float64))
            inside = stl_inside_mask(fld, turned if turned is not None else surf)
            n, n_stl = np.count_nonzero(inside), np.count_nonzero(self.static)
            if self.static is not None and abs(n - n_stl) > 0.2 * n_stl:
                raise ValueError(f"{n:,} cells inside it, {n_stl:,} inside the STL")
            return inside
        except Exception as e:
            warn(f"posed moving-body mesh not usable ({e}); using '{self.model.impeller_stl.name}' "
                 f"in its initial pose" if self.model.impeller_stl is not None else
                 f"posed moving-body mesh not usable ({e}); moving body not excluded")
            self.posed = False
            self.name = self.model.impeller_stl.name if self.model.impeller_stl is not None else ""
            return self.static


def load_field(frames: list[tuple[float, Path]], var_name: str, vm_name: str | None,
               body: MovingBody | None = None, shared: dict | None = None) -> Field3D:
    """The field of one frame as it is, or the time average over several frames.

    Averaging keeps running sums, so only one frame is in memory at a time. A cell is
    fluid in a frame when its velocity magnitude is > 0 and, with a posed moving-body
    mesh, it is not inside the body at that time. It is fluid when that holds in at
    least half of the frames; its value is the mean over the frames in which it was
    fluid. The mean velocity magnitude is set to 0 in the other cells, so 'velocity
    magnitude > 0' stays the fluid test.

    `shared` is kept between the variables of one run for the same frames: it holds
    the velocity magnitude, the body cells and the per-frame fluid flags of the first
    variable, so the next one reads only its own array.
    """
    import pyvista as pv

    reuse = bool(shared) and "velmag" in shared
    names = [var_name] if reuse else list(dict.fromkeys(n for n in (var_name, vm_name) if n))
    n_frames = len(frames)
    fld = None
    flags, solid, solid_count, posed = [], None, None, False
    for k, (t, path) in enumerate(frames):
        nc, origin, spacing, arrays, covered = read_frame(path, names)
        var = arrays[var_name]
        if fld is None:
            fld = Field3D(grid=None, nc=nc, origin=origin, spacing=spacing, var_name=var_name,
                          var=var, velmag=None, covered=covered)
            if n_frames > 1:
                sum_var = np.zeros(var.shape, np.float64)
                sum_vm = np.zeros(var.shape, np.float64) if vm_name and not reuse else None
                count = np.zeros(var.shape, np.int32)
        elif nc != fld.nc:
            fail(f"grid size changes between frames ({fld.nc} vs {nc} cells in {path.name})")
        if reuse:
            if n_frames == 1:
                break
            is_fluid = np.unpackbits(shared["flags"][k], count=var.size).reshape(var.shape).view(bool)
        else:
            velmag = arrays.get(vm_name)
            if body is not None and (body.posed or n_frames == 1):
                solid = body.mask(fld, t)
                if posed and not body.posed:
                    fail(f"no usable posed moving-body mesh for {path.name}")
                posed = body.posed
            if n_frames == 1:
                break
            is_fluid = velmag > 0.0 if velmag is not None else np.isfinite(var)
            if posed and solid is not None:
                is_fluid &= ~solid
                if solid_count is None:
                    solid_count = np.zeros(var.shape, np.int32)
                solid_count += solid
            if velmag is not None:
                np.add(sum_vm, velmag, out=sum_vm, where=is_fluid)
            if shared is not None:
                flags.append(np.packbits(is_fluid))
        np.add(sum_var, var, out=sum_var, where=is_fluid)
        count += is_fluid
    if n_frames > 1:
        seen = np.maximum(count, 1)
        var = (sum_var / seen).astype(np.float32)
        if not reuse:
            if velmag is not None:
                velmag = np.where(2 * count >= n_frames, sum_vm / seen, 0.0).astype(np.float32)
            if solid_count is not None:  # posed meshes: inside the body in half of the frames
                solid = 2 * solid_count >= n_frames
            elif body is not None:
                solid = body.mask(fld, frames[-1][0])
        del sum_var, sum_vm, seen, count
    if reuse:
        velmag, solid, fld.body = shared["velmag"], shared["solid"], shared["body"]
    else:
        if body is not None and solid is not None:
            fld.body = ((f"posed mesh {body.name} of " + ("the frame" if n_frames == 1 else
                                                           f"each of the {n_frames} frames"))
                        if body.posed else f"'{body.name}' in its initial pose")
        if shared is not None and velmag is not None:  # the flags do not depend on the variable
            shared.update(velmag=velmag, solid=solid, body=fld.body, flags=flags)

    fld.grid = pv.ImageData(dimensions=tuple(n + 1 for n in nc), origin=tuple(origin),
                            spacing=tuple(spacing))
    fld.var, fld.velmag, fld.solid = var, velmag, solid
    return fld


# ---------------------------------------------------------------- masks

def stl_inside_mask(fld: Field3D, stl) -> np.ndarray:
    """Cells whose centers lie inside a closed surface (an STL file or a PolyData)."""
    import pyvista as pv

    shape = (fld.nc[2], fld.nc[1], fld.nc[0])
    surf = pv.read(str(stl)) if isinstance(stl, (str, Path)) else stl
    try:  # rasterise the closed surface onto the cell-centre grid: seconds instead of minutes
        from vtkmodules.util.numpy_support import vtk_to_numpy
        from vtkmodules.vtkImagingStencil import vtkImageStencilToImage, vtkPolyDataToImageStencil

        stencil = vtkPolyDataToImageStencil()
        stencil.SetInputData(surf)
        stencil.SetOutputOrigin(*(fld.origin + 0.5 * fld.spacing))
        stencil.SetOutputSpacing(*fld.spacing)
        stencil.SetOutputWholeExtent(0, fld.nc[0] - 1, 0, fld.nc[1] - 1, 0, fld.nc[2] - 1)
        image = vtkImageStencilToImage()
        image.SetInputConnection(stencil.GetOutputPort())
        image.SetInsideValue(1)
        image.SetOutsideValue(0)
        image.SetOutputScalarTypeToUnsignedChar()
        image.Update()
        inside = vtk_to_numpy(image.GetOutput().GetPointData().GetScalars())
        return inside.reshape(shape).astype(bool)
    except Exception as e:
        warn(f"image-stencil inside test failed ({e}); using the enclosed-points test")
    X, Y, Z = fld.cell_centers_3d()
    pts = pv.PolyData(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
    try:  # pyvista >= 0.46; select_enclosed_points is deprecated
        sel = pts.select_interior_points(surf, check_surface=False)
        inside = np.asarray(sel["selected_points"])
    except AttributeError:
        sel = pts.select_enclosed_points(surf, check_surface=False)
        inside = sel["SelectedPoints"].astype(bool)
    return inside.reshape(shape)


def build_fluid_mask(fld: Field3D, fluid_stl: Path | None, shared: dict | None = None) -> np.ndarray:
    """Fluid cells: finite value, non-zero velocity magnitude, not inside the moving body
    (fld.solid). `shared` keeps the mask for the next variable of the same frames."""
    finite = np.isfinite(fld.var)
    if shared and "fluid" in shared:
        log("fluid mask: reused from the previous variable (same frames)")
        return shared["fluid"] if finite.all() else shared["fluid"] & finite
    # solid/exterior cells in M-Star volume output carry exact zeros
    valid = finite.copy()
    if fld.covered is not None:
        valid &= fld.covered
    by_velocity = fld.velmag is not None and bool((fld.velmag > 0.0).any())
    if by_velocity:
        valid &= fld.velmag > 0.0
        log(f"solid/exterior blanking via zero velocity magnitude: "
            f"{np.count_nonzero(~valid):,} of {valid.size:,} cells excluded")
    if fld.solid is not None:
        # moving-body cells carry the solid's velocity, so zero-blanking misses them
        valid &= ~fld.solid
        log(f"moving body excluded ({fld.body}): {np.count_nonzero(fld.solid):,} cells")
    if by_velocity:
        # the fill STL is only the initial region of one fluid: it is not used here
        log("fluid mask: non-zero velocity magnitude"
            + (f" (initial-fill '{fluid_stl.name}' not used)" if fluid_stl is not None else ""))
        log(f"fluid cells: {np.count_nonzero(valid):,} "
            f"({100 * np.count_nonzero(valid) / valid.size:.1f}% of domain)")
    elif fluid_stl is not None:
        valid = _fill_stl_mask(fld, fluid_stl, valid)
    if shared is not None and finite.all():  # else the mask depends on this variable
        shared["fluid"] = valid
    return valid


def _fill_stl_mask(fld: Field3D, fluid_stl: Path, valid: np.ndarray) -> np.ndarray:
    """Fallback without a usable velocity array: crop to the initial-fill STL."""
    import pyvista as pv

    warn("no usable velocity magnitude array: fluid mask falls back to the initial-fill STL")
    mesh = pv.read(str(fluid_stl))
    b = np.array(mesh.bounds).reshape(3, 2)
    boxy = False
    try:
        bounds_vol = np.prod(b[:, 1] - b[:, 0])
        boxy = bounds_vol > 0 and abs(mesh.volume - bounds_vol) / bounds_vol < 0.05
    except Exception:
        pass
    if boxy:
        xc, yc, zc = fld.cell_centers_1d()
        inside = (((xc >= b[0, 0]) & (xc <= b[0, 1]))[None, None, :] &
                  ((yc >= b[1, 0]) & (yc <= b[1, 1]))[None, :, None] &
                  ((zc >= b[2, 0]) & (zc <= b[2, 1]))[:, None, None])
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
    xc, yc, zc = fld.cell_centers_1d()
    d = (xc[None, None, :] - center[0], yc[None, :, None] - center[1], zc[:, None, None] - center[2])
    k = int(np.argmax(np.abs(axis)))
    if abs(axis[k]) == 1.0 and np.count_nonzero(axis) == 1:
        # shaft along a grid direction: the same sums, without 3-D temporaries
        i, j = (m for m in range(3) if m != k)
        axial = d[k] * axis[k]
        rad2 = d[i] ** 2 + d[j] ** 2
    else:
        axial = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2]
        rad2 = ((d[0] - axial * axis[0]) ** 2 + (d[1] - axial * axis[1]) ** 2 +
                (d[2] - axial * axis[2]) ** 2)
    return (rad2 <= radius ** 2) & (axial >= span[0]) & (axial <= span[1])


def detect_impeller_zone(fld: Field3D, fluid: np.ndarray, model: ModelInfo,
                         pad: float, percentile: float, shape: str) -> np.ndarray:
    from scipy import ndimage

    if model.impeller_stl is not None and model.impeller_axis is not None:
        mesh = load_impeller(model)
        axis = model.impeller_axis
        center = np.array(mesh.center)
        if model.impeller_pivot is not None:
            # on the rotation axis (the line through the pivot), level with the STL centre
            center = model.impeller_pivot + ((center - model.impeller_pivot) @ axis) * axis
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
    X, Y, Z = (fld.coordinate(i, core) for i in range(3))  # centres of the core cells
    axis = np.array([0.0, 1.0, 0.0])  # default tank axis when nothing else is known
    center = np.array([X.mean(), Y.mean(), Z.mean()])
    dxyz = np.stack([X - center[0], Y - center[1], Z - center[2]])
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
    bounds = log_base ** (uniq[1:] + 1.0)  # upper edge of each bin below the top one
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
            # merge into another value class, never into the geometric impeller zone
            neigh = neigh[(neigh >= first_mergeable) & (neigh != c)]
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

def _stable_order(labels: np.ndarray) -> np.ndarray:
    """argsort(labels, kind='stable') in linear time (numpy radix-sorts 16-bit keys)."""
    order = np.argsort(labels.astype(np.uint16), kind="stable")  # low 16 bits
    if labels.max() > 0xFFFF:
        order = order[np.argsort((labels[order] >> 16).astype(np.uint16), kind="stable")]
    return order


def collect_zone_rows(zone_class: np.ndarray, zone_id: np.ndarray, fluid: np.ndarray,
                      var: np.ndarray, cell_vol: float) -> list[tuple[str, int, str, dict]]:
    """Per-class and per-subzone stats rows: (level, id, label, summary)."""
    n_fluid = int(np.count_nonzero(fluid))
    labelled = zone_class > 0
    cls, ids, vals = zone_class[labelled], zone_id[labelled], var[labelled]  # grid order

    def summary(v: np.ndarray) -> dict:
        n = np.int64(v.size)
        return {
            "cells": int(n),
            "volume_m3": n * cell_vol,
            "vol_frac": n / n_fluid,
            "mean": float(v.mean()) if v.size else np.nan,
            "std": float(v.std()) if v.size else np.nan,
            "min": float(v.min()) if v.size else np.nan,
            "max": float(v.max()) if v.size else np.nan,
        }

    rows = []
    for level, labels in (("class", cls), ("subzone", ids)):
        counts = np.bincount(labels) if labels.size else np.zeros(1, np.int64)
        present = np.flatnonzero(counts)
        present = present[present >= 1]
        if present.size > 8:  # many groups: sort once, each group keeps its grid order
            order = _stable_order(labels)
            v_sorted, c_sorted = vals[order], cls[order]
            ends = np.cumsum(counts)
        for z in present:
            if present.size > 8:
                a, b = ends[z] - counts[z], ends[z]
                v, c = v_sorted[a:b], int(c_sorted[a])
            else:
                mask = labels == z
                v, c = vals[mask], int(cls[np.argmax(mask)])
            rows.append((level, int(z), zone_label(c), summary(v)))
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
    qs = (5, 25, 50, 75, 95, 99)
    pct = {f"p{q:02d}": float(p) for q, p in zip(qs, np.percentile(v, qs))}

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
    if "mstar_global_mean" in het:
        rsd = (f"   RSD = {het['mstar_rsd_percent']:.3f}%" if "mstar_rsd_percent" in het else "")
        print(f"  M-Star scalar stats: global mean = {het['mstar_global_mean']:.6g}{rsd} "
              f"at t = {het['mstar_stats_time_s']:.3g} s")
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

def view_center(model: ModelInfo, centers: tuple, zone_class: np.ndarray) -> np.ndarray:
    """The point the side views are cut through: on the impeller axis (the pivot of
    input.xml, else the centre of the STL), or the centroid of the labelled cells when
    there is no impeller. centers = cell centres along (x, y, z)."""
    if model.impeller_pivot is not None:
        return model.impeller_pivot
    if model.impeller_stl is not None:
        return np.array(load_impeller(model).center)
    lab = zone_class > 0
    along = (lab.sum(axis=(0, 1)), lab.sum(axis=(0, 2)), lab.sum(axis=(1, 2)))  # x, y, z
    return np.array([float((cc * n).sum() / max(n.sum(), 1)) for cc, n in zip(centers, along)])


def zone_views(fld: Field3D, zone_class: np.ndarray, axis: np.ndarray | None,
               center: np.ndarray | None = None) -> list[tuple]:
    """ZoneClass cross-sections: two side views through `center` (a point on the tank
    axis, see view_center; coordinate 0 when not given; not the grid mid-index: the
    domain need not be centered) and a top view normal to the shaft through the
    impeller-zone centre.
    [(name, img[v, u], (u_centres, v_centres), u_label, v_label, description, plane_coord)]"""
    xc, yc, zc = fld.cell_centers_1d()
    u = np.abs(axis) if axis is not None else np.array([0.0, 1.0, 0.0])  # dominant axis only
    c = np.zeros(3) if center is None else center
    kx, ky, kz = (int(np.argmin(np.abs(cc - c[i]))) for i, cc in enumerate((xc, yc, zc)))
    if u[2] < 0.9:  # slice normal to z, plus perpendicular side view normal to x
        views = [
            ("side", zone_class[kz, :, :], (xc, yc), "x [m]", "y [m]",
             "side view, facing +Z", float(zc[kz])),
            ("side2", zone_class[:, :, kx].T, (zc, yc), "z [m]", "y [m]",
             "side view, facing +X", float(xc[kx])),
        ]
    else:  # z-axis tank: slice normal to y, plus perpendicular side view normal to x
        views = [
            ("side", zone_class[:, ky, :], (xc, zc), "x [m]", "z [m]",
             "side view, facing +Y", float(yc[ky])),
            ("side2", zone_class[:, :, kx], (yc, zc), "y [m]", "z [m]",
             "side view, facing +X", float(xc[kx])),
        ]
    ax_xyz = int(np.argmax(u))
    np_ax = 2 - ax_xyz  # zone_class axes are [z, y, x]
    imp_idx = np.nonzero(zone_class == 1)[np_ax]
    k = int(round(imp_idx.mean())) if imp_idx.size else zone_class.shape[np_ax] // 2
    if ax_xyz == 1:
        views.append(("top", zone_class[:, k, :], (xc, zc), "x [m]", "z [m]",
                      f"top view, y = {yc[k]:.4f} m", float(yc[k])))
    elif ax_xyz == 2:
        views.append(("top", zone_class[k, :, :], (xc, yc), "x [m]", "y [m]",
                      f"top view, z = {zc[k]:.4f} m", float(zc[k])))
    else:
        views.append(("top", zone_class[:, :, k], (yc, zc), "y [m]", "z [m]",
                      f"top view, x = {xc[k]:.4f} m", float(xc[k])))
    return views


def write_zone_slices(out_dir: Path, fld: Field3D, zone_class: np.ndarray,
                      axis: np.ndarray | None, center: np.ndarray | None = None,
                      max_cells: int = 400) -> None:
    """The zone maps as data: zone_slices.csv (one row per view with its grid) and
    zone_slice_<view>.csv (ZoneClass matrix, rows = v, columns = u, first column = v
    coordinate, header = u coordinates; 0 = solid/outside), strided to <= max_cells
    per axis so a viewer can draw many cases at once."""
    meta = []
    for name, img, (uc, vc), ul, vl, desc, coord in zone_views(fld, zone_class, axis, center):
        s = max(1, -(-max(img.shape) // max_cells))
        img, uc, vc = img[::s, ::s], uc[::s], vc[::s]
        du = float(uc[1] - uc[0]) if len(uc) > 1 else s * float(fld.spacing["xyz".index(ul[0])])
        dv = float(vc[1] - vc[0]) if len(vc) > 1 else s * float(fld.spacing["xyz".index(vl[0])])
        with open(out_dir / f"zone_slice_{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([f"{vl[0]} \\ {ul[0]} [m]", *(f"{x:.6g}" for x in uc)])
            for v, row in zip(vc, img):
                w.writerow([f"{v:.6g}", *row.tolist()])
        meta.append([name, desc, ul, vl, f"{uc[0]:.6g}", f"{du:.6g}", len(uc),
                     f"{vc[0]:.6g}", f"{dv:.6g}", len(vc), f"{coord:.6g}",
                     fld.var_name, var_unit(fld.var_name).strip()])
    with open(out_dir / "zone_slices.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["view", "title", "u_label", "v_label", "u0 [m]", "du [m]", "nu [count]",
                    "v0 [m]", "dv [m]", "nv [count]", "plane_coordinate [m]", "variable", "unit"])
        w.writerows(meta)
    log(f"wrote zone slices to {out_dir}")


def write_zoner_summary(out_dir: Path, case_name: str, variable: str, fld: Field3D,
                        het: dict, class_means: dict[int, float], t_sel: float,
                        t_steady: float | None, t_avg: float | None = None,
                        n_frames: int = 1) -> None:
    """zoner_summary.csv: one row of scalar results (column 'case' first) so a viewer
    can compare cases; column names carry the requested variable and units."""
    unit = var_unit(fld.var_name).strip() or "1"
    ratio = f"({unit})/({unit})"
    a = variable.strip()
    cols = {
        "case": case_name,
        f"variable ({a})": fld.var_name,
        f"analysis time ({a}) [s]": t_sel,
        f"steady start ({a}) [s]": "" if t_steady is None else t_steady,
        f"averaging start ({a}) [s]": t_sel if t_avg is None else t_avg,
        f"frames averaged ({a}) [count]": n_frames,
        f"zones ({a}) [count]": len(class_means),
        f"global mean ({a}) [{unit}]": het.get("global_mean"),
        f"CV ({a}) [{ratio}]": het.get("cv"),
        f"sigma log10 ({a}) [log10({unit})]": het.get("sigma_log10"),
        f"p50 ({a}) [{unit}]": het.get("p50"),
        f"p95 ({a}) [{unit}]": het.get("p95"),
        f"p99 ({a}) [{unit}]": het.get("p99"),
        f"p95/p50 ({a}) [{ratio}]": het.get("p95_over_p50"),
        f"p99/mean ({a}) [{ratio}]": het.get("p99_over_mean"),
        f"eta2 between zones ({a}) [{ratio}]": het.get("eta2_between_zone"),
        f"zone contrast ({a}) [{ratio}]": het.get("zone_contrast"),
        f"gradient index ({a}) [{ratio}]": het.get("grad_index"),
        f"gradient index p95 ({a}) [{ratio}]": het.get("grad_index_p95"),
        f"significant gradients ({a}) [bool]": int(bool(het.get("significant_gradients"))),
    }
    if "mstar_global_mean" in het:
        cols[f"M-Star global mean ({a}) [{unit}]"] = het["mstar_global_mean"]
    if "mstar_rsd_percent" in het:
        cols[f"M-Star RSD ({a}) [%]"] = het["mstar_rsd_percent"]
    with open(out_dir / "zoner_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols.keys())
        w.writerow(cols.values())
    log(f"wrote {out_dir / 'zoner_summary.csv'}")

def log_edges(pos: np.ndarray, n: int) -> np.ndarray:
    """n log-spaced histogram edges that include the smallest and the largest value
    (the rounded end points of logspace can miss them)."""
    edges = np.logspace(np.log10(pos.min()), np.log10(pos.max()), n)
    edges[0], edges[-1] = min(edges[0], pos.min()), max(edges[-1], pos.max())
    return edges


def write_data_csvs(out_dir: Path, fld: Field3D, fluid: np.ndarray, zone_class: np.ndarray,
                    class_upper: dict[int, float | None], trace: tuple | None,
                    t_steady: float | None, t_sel: float, axis: np.ndarray | None,
                    class_means: dict[int, float] | None = None) -> None:
    """The data behind the diagnostic plots as CSVs, so a viewer can draw them:
    time_trace.csv, histogram.csv (log bins), class_boundaries.csv (upper boundary of
    each value class), exposure_cdf.csv (volume fraction at or below a value),
    zone_weights.csv and profiles.csv (axial / radial volume-averaged profiles)."""
    unit = var_unit(fld.var_name)
    if trace is not None:
        times, values, label = trace
        with open(out_dir / "time_trace.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["time [s]", label, "steady_start_s", "analysis_time_s"])
            for i, (t, v) in enumerate(zip(times, values)):
                w.writerow([t, v, t_steady if i == 0 else "", t_sel if i == 0 else ""])
    vals = fld.var[fluid].astype(np.float64)
    pos = vals[vals > 0]
    if pos.size:
        edges = log_edges(pos, 121)
        counts, _ = np.histogram(pos, edges)
        with open(out_dir / "histogram.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([f"bin_low [{unit.strip()}]", f"bin_high [{unit.strip()}]", "cells [count]", "volume_fraction [m^3/m^3]"])
            for lo, hi, n in zip(edges[:-1], edges[1:], counts):
                w.writerow([lo, hi, int(n), n / vals.size])
    sv = np.sort(vals)
    idx = np.linspace(0, sv.size - 1, min(4000, sv.size)).astype(int)
    with open(out_dir / "exposure_cdf.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([fld.var_name, "cumulative_volume_fraction [m^3/m^3]"])
        for i in idx:
            w.writerow([sv[i], (i + 1) / sv.size])
    with open(out_dir / "class_boundaries.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "label", f"zone_mean [{unit.strip()}]", f"upper_boundary [{unit.strip()}]"])
        for c in sorted(class_means or {}):  # blank: impeller zone and the top value class
            upper = class_upper.get(c)
            w.writerow([c, zone_label(c, class_means, unit), class_means[c],
                        "" if upper is None else upper])
    if class_means:
        cls = sorted(class_means)
        n_fluid = np.count_nonzero(fluid)
        vfrac = np.array([np.count_nonzero(zone_class == c) / n_fluid for c in cls])
        weight = vfrac * np.array([class_means[c] for c in cls])
        weight /= weight.sum() if weight.sum() else 1.0
        with open(out_dir / "zone_weights.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["class", "label", "volume_fraction [m^3/m^3]", f"contribution_to_mean [({unit.strip()})/({unit.strip()})]"])
            for c, vf, wt in zip(cls, vfrac, weight):
                w.writerow([c, zone_label(c, class_means, unit), vf, wt])
    u = np.abs(axis) if axis is not None else np.array([0.0, 1.0, 0.0])  # dominant axis only
    up_i = int(np.argmax(u))
    a = fld.coordinate(up_i, fluid).astype(np.float64)
    others = [fld.coordinate(i, fluid).astype(np.float64) for i in range(3) if i != up_i]
    r = np.hypot(others[0] - others[0].mean(), others[1] - others[1].mean())
    with open(out_dir / "profiles.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["profile", "coordinate [m]", f"{fld.var_name} (volume mean)", "cells [count]"])
        for name, coord in ((f"axial ({'xyz'[up_i]})", a), ("radial", r)):
            edges = np.linspace(coord.min(), coord.max(), 41)
            sums, _ = np.histogram(coord, edges, weights=vals)
            cnts, _ = np.histogram(coord, edges)
            for mid, s, n in zip((edges[:-1] + edges[1:]) / 2, sums, cnts):
                w.writerow([name, mid, s / n if n else "", int(n)])
    log(f"wrote plot data CSVs to {out_dir}")


def make_plots(out_dir: Path, fld: Field3D, fluid: np.ndarray, zone_class: np.ndarray,
               class_upper: dict[int, float | None], trace: tuple | None,
               t_steady: float | None, t_sel: float, axis: np.ndarray | None,
               class_means: dict[int, float] | None = None,
               center: np.ndarray | None = None) -> None:
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
        ax.hist(pos, bins=log_edges(pos, 120), color="steelblue", alpha=0.8)
        ax.set_xscale("log")
    for i, b in enumerate(b for b in class_upper.values() if b is not None):
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

    for name, img, (uc, vc), xl, yl, desc, _ in zone_views(fld, zone_class, axis, center):
        extent = [uc[0], uc[-1], vc[0], vc[-1]]
        suf = "" if name == "side" else f"_{name}"
        plot_slice(img, extent, xl, yl, f"Zone map ({desc})", f"zone_slice{suf}.png")
        if class_means:
            plot_pct_slice(img, extent, xl, yl,
                           f"Zone mean as % of global mean ({desc})",
                           f"zone_pct_slice{suf}.png")

    # volume-exposure curve: interactive Plotly HTML (click legend to show/hide)
    import plotly.graph_objects as go
    sv = np.sort(vals.astype(np.float64))
    idx = np.linspace(0, sv.size - 1, min(4000, sv.size)).astype(int)
    pfig = go.Figure()
    pfig.add_trace(go.Scatter(x=sv[idx], y=100 * (idx + 1) / sv.size, mode="lines",
                              name="fluid volume CDF",
                              line=dict(color="#4682b4", width=2)))

    def add_vline(x, name, color, dash):
        # vertical lines as traces (not shapes) so they appear in / toggle from the legend
        pfig.add_trace(go.Scatter(x=[x, x], y=[0, 100], mode="lines", name=name,
                                  line=dict(color=color, dash=dash, width=1.5)))

    add_vline(overall_mean, f"global mean ({overall_mean:.3g}{unit})", "#000000", "dash")
    if class_means:
        for c in sorted(class_means):
            add_vline(class_means[c], zone_label(c, class_means, unit), _hex(colors[c]), "dot")
    pfig.update_layout(template="simple_white", width=950, height=520,
                       title="Volume exposure: fraction of fluid at or below a value",
                       xaxis_title=fld.var_name,
                       yaxis_title="cumulative fluid volume [%]",
                       legend=dict(font=dict(size=10)))
    if (sv > 0).any():
        pfig.update_xaxes(type="log")
    write_interactive_html(pfig, out_dir / "exposure_cdf.html")

    # axial & radial profiles: where the gradients sit macroscopically
    up_i = int(np.argmax(u))
    a = fld.coordinate(up_i, fluid).astype(np.float64)
    others = [fld.coordinate(i, fluid).astype(np.float64) for i in range(3) if i != up_i]
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
        pl.add_mesh(load_impeller(model), color="dimgray")
    pl.add_legend(bcolor="white", size=(0.32, 0.18), loc="upper left")
    up = tuple(up_axis(model.impeller_axis))
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

def run_case(case: Case, model: ModelInfo, variable: str, args, out_dir: Path,
             shared: dict | None = None) -> CaseResult:
    """Full zoning/analysis pipeline for one case and one variable. `shared` (one dict
    per case) carries what does not depend on the variable to the next variable."""
    trace = steady_trace_from_stats(case, variable, args.steady_on)
    t_steady = None
    no_plateau = False
    if args.time is not None:
        t_target = args.time
        log(f"user-defined start time: {t_target} s")
    elif trace is None:
        warn("no stats trace available; using last VTI timestep")
        t_target = case.vti_series[-1][0]
    else:
        t_steady = detect_steady_state(trace[0], trace[1], args.window, args.tol)
        if t_steady is not None and trace[0][-1] - t_steady < 2 * args.window:
            t_steady = None  # shorter than the two windows the test compares: not a plateau
        if t_steady is None:  # fall back to the last 25 % of the trace's time span
            no_plateau = True
            t_target = float(f"{trace[0][-1] - 0.25 * (trace[0][-1] - trace[0][0]):.6g}")
        else:
            t_target = t_steady
            log(f"steady state detected at t = {t_steady:.3f} s")

    # --- frames to analyse: one frame, the Time-Avg array of the last frame, or the
    # average of the instantaneous field over every frame from t_target on
    names = scalar_cell_arrays(case.vti_series[-1][1])
    var_name = resolve_variable(variable, names, prefer_time_avg=not args.instantaneous)
    vm = [n for n in names if norm_name(n) == "velocity magnitude"]
    later = [(t, f) for t, f in case.vti_series if t >= t_target - 1e-9] or case.vti_series[-1:]
    if args.instantaneous:
        frames = later[-1:] if no_plateau else later[:1]
    elif "time avg" in norm_name(var_name):
        frames = case.vti_series[-1:]
        if args.time is not None:
            warn("--time does not select the frame of a Time-Avg array (last frame is used)")
    else:
        frames = later
    if no_plateau:
        warn(f"no steady plateau of at least two windows found (window={args.window}s, tol={args.tol}): "
             + ("using the last frame" if frames is not later else
                f"using the frames of the last 25 % of the trace (t >= {t_target:.3f} s)"))
    t_avg, t_sel = frames[0][0], frames[-1][0]
    if len(frames) > 1:
        log(f"time-averaged over {len(frames)} frames, t = {t_avg:.3g} .. {t_sel:.3g} s")
    else:
        log(f"analysis timestep: t = {t_sel:.3g} s ({frames[0][1].name})"
            + ("" if args.instantaneous or frames[0][1] != case.vti_series[-1][1] else ", last frame"))
    log(f"analysis variable: '{var_name}'")

    # --- load field & masks
    same = None  # velocity magnitude, body cells and masks of these frames
    if shared is not None:
        same = shared.setdefault(tuple(f for _, f in frames), {})
        body = shared.setdefault("body", MovingBody(case, model))
    else:
        body = MovingBody(case, model)
    fld = load_field(frames, var_name, vm[0] if vm else None, body, same)
    fluid = build_fluid_mask(fld, model.fluid_stl, same)
    if not fluid.any():
        fail("fluid mask is empty")

    if not HAS_IMPELLER:
        impeller = np.zeros(fld.var.shape, bool)
        log("impeller zone disabled; all zones from clustering")
    elif same and same.get("fluid") is fluid and "impeller" in same:
        impeller = same["impeller"]
        log(f"impeller zone: reused from the previous variable, {np.count_nonzero(impeller):,} cells")
    else:
        impeller = detect_impeller_zone(fld, fluid, model, args.impeller_pad,
                                        args.impeller_percentile, args.impeller_shape)
        if same is not None and same.get("fluid") is fluid:
            same["impeller"] = impeller

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
    # upper boundary of each value class: none for the impeller zone and the top class
    class_upper = {c: (float(bounds[c - class_offset - 1])
                       if 0 < c - class_offset <= len(bounds) else None) for c in class_means}
    center = view_center(model, fld.cell_centers_1d(), zone_class)
    rows = collect_zone_rows(zone_class, zone_id, fluid, fld.var, cell_vol)
    report(rows, fld.var_name, out_dir / "zones.csv")
    het = heterogeneity_stats(fld, fluid, zone_class, model,
                              args.eta2_threshold, args.contrast_threshold)
    scalar_stats = scalar_stats_at_time(case, variable, t_sel)
    if scalar_stats is not None:
        het["mstar_stats_time_s"] = scalar_stats["time_s"]
        het["mstar_global_mean"] = scalar_stats["mean"]
        if "rsd_percent" in scalar_stats:
            het["mstar_rsd_percent"] = scalar_stats["rsd_percent"]
        log(f"scalar stats from {scalar_stats['source']} at t={scalar_stats['time_s']:.3g} s: "
            f"mean={scalar_stats['mean']:.6g}"
            + (f", RSD={scalar_stats['rsd_percent']:.3g}%"
               if "rsd_percent" in scalar_stats else ""))
    report_heterogeneity(het, fld.var_name, out_dir / "heterogeneity.csv",
                         args.eta2_threshold, args.contrast_threshold)
    write_labeled_vti(fld, zone_class, zone_id, out_dir / "zones.vti")
    write_data_csvs(out_dir, fld, fluid, zone_class, class_upper, trace, t_steady, t_sel,
                    model.impeller_axis, class_means)
    write_zone_slices(out_dir, fld, zone_class, model.impeller_axis, center)
    write_zoner_summary(out_dir, case.root.name, variable, fld, het, class_means, t_sel, t_steady,
                        t_avg, len(frames))
    if not args.no_plots:
        make_plots(out_dir, fld, fluid, zone_class, class_upper, trace, t_steady, t_sel,
                   model.impeller_axis, class_means, center)
    if not args.no_render:
        render_3d(out_dir, fld, zone_class, model, show=args.show, html=args.html,
                  class_means=class_means)
    global_mean = (scalar_stats["mean"] if scalar_stats is not None
                   else float(fld.var[fluid].mean()))
    return CaseResult(case_name=case.root.name, var_name=fld.var_name, t_sel=t_sel,
                      global_mean=global_mean,
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
    c = view_center(r.model, (xc, yc, zc), zone_class)
    if u[2] < 0.9:  # y-up tank: slice normal to z through the axis (not the grid mid-index)
        k = int(np.argmin(np.abs(zc - c[2])))
        img = zone_class[k, :, :]
        extent = [xc[0], xc[-1], yc[0], yc[-1]]
        xl, yl = "x [m]", "y [m]"
    else:  # z-up tank: slice normal to y through the axis
        k = int(np.argmin(np.abs(yc - c[1])))
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


def _hex(rgba) -> str:
    """Matplotlib RGBA floats (0-1) -> '#rrggbb' for Plotly / HTML color inputs."""
    return "#{:02x}{:02x}{:02x}".format(*(int(round(255 * v)) for v in rgba[:3]))


def write_interactive_html(fig, out_path: Path, color_pickers: bool = True) -> None:
    """Standalone Plotly HTML: click legend to show/hide, double-click to isolate,
    plus an optional per-series color picker row below the plot."""
    html = fig.to_html(
        full_html=True,
        include_plotlyjs=True,  # embeds plotly.js (~3.5 MB) so it works offline; "cdn" = small file
        div_id="plot",
        config={"editable": True, "displaylogo": False,
                "toImageButtonOptions": {"format": "svg", "filename": out_path.stem}},
    )
    if color_pickers:
        picker_js = """
<div id="pickers" style="font:12px sans-serif;padding:8px 16px"></div>
<script>
window.addEventListener('load', () => {
  const gd = document.getElementById('plot');
  const box = document.getElementById('pickers');
  gd.data.forEach((tr, i) => {
    if (!tr.line || !tr.line.color) return;
    const lab = document.createElement('label');
    lab.style.cssText = 'margin-right:14px;white-space:nowrap';
    const inp = document.createElement('input');
    inp.type = 'color';
    inp.value = tr.line.color;
    inp.oninput = e => Plotly.restyle(gd, {'line.color': e.target.value}, [i]);
    lab.append(inp, ' ' + tr.name);
    box.append(lab);
  });
});
</script>
"""
        html = html.replace("</body>", picker_js + "</body>")
    out_path.write_text(html, encoding="utf-8")
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
    """Combined volume-exposure curves of all cases as interactive Plotly HTML
    (values reloaded from zones.vti). Click legend entries to show/hide."""
    import plotly.graph_objects as go
    from plotly.colors import qualitative

    palette = qualitative.Plotly  # hex colors; swap for qualitative.D3, .Set1, etc.
    fig = go.Figure()
    any_pos = False
    for i, r in enumerate(results):
        grid, zone_class, _ = load_zone_class(r)
        var = np.asarray(grid.cell_data[r.var_name]).reshape(zone_class.shape)
        v = np.sort(var[zone_class > 0].astype(np.float64))
        any_pos |= bool((v > 0).any())
        idx = np.linspace(0, v.size - 1, min(4000, v.size)).astype(int)
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(x=v[idx], y=100 * (idx + 1) / v.size, mode="lines",
                                 name=r.case_name, legendgroup=r.case_name,
                                 line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(x=[r.global_mean] * 2, y=[0, 100], mode="lines",
                                 name=f"{r.case_name} global mean ({r.global_mean:.3g})",
                                 legendgroup=r.case_name,
                                 line=dict(color=color, dash="dash", width=1)))
    fig.update_layout(template="simple_white", width=950, height=560,
                      title="Volume exposure by case (dashed: case global means)",
                      xaxis_title=results[0].var_name,
                      yaxis_title="cumulative fluid volume [%]",
                      legend=dict(font=dict(size=10), groupclick="toggleitem"))
    if any_pos:
        fig.update_xaxes(type="log")
    write_interactive_html(fig, out_path)


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
                pl.add_mesh(load_impeller(r.model), color="dimgray")
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
        up = tuple(up_axis(r.model.impeller_axis))
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
    global_mean = float(het.get("mstar_global_mean", var[zone_class > 0].mean()))

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
        shared = {} if len(args.variable) > 1 else None
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
                    results[v].append(run_case(case, model, v, args, out_dir, shared))
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
        batch_exposure_cdf(results[v], comp_dir / "exposure_cdf.html")
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
                    help="start of the averaging window [s] (frames from this time to the "
                         "last one are averaged); with --instantaneous the analysed frame "
                         "is the first one at or after it; overrides steady-state "
                         "detection; no effect on a Time-Avg array (last frame)")
    ap.add_argument("--steady-on", choices=["variable", "power"], default="variable",
                    help="steady-state criterion source")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="steady-state tolerance on windowed-mean drift")
    ap.add_argument("--window", type=float, default=1.0, help="steady-state window [s]")
    ap.add_argument("--instantaneous", action="store_true",
                    help="analyse one instantaneous frame (the first at or after the steady "
                         "start or --time) instead of the Time-Avg array of the last frame "
                         "or, without a Time-Avg variant, the average over all frames from "
                         "the steady start on")
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
    shared = {} if len(args.variable) > 1 else None  # what the variables have in common
    for v in args.variable:
        if len(args.variable) > 1:
            log(f"--- variable: {v} ---")
        run_case(case, model, v, args, case_out_dir(base, args.variable, v), shared)
    log("done")


if __name__ == "__main__":
    main()
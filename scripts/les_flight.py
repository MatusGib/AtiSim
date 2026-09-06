"""Fly AtiSim through Yoshimura's LES field -- the first non-circular field.

EVERY load comparison in PROJECT.md section 4 is flown through a wind field that
was identified FROM the accelerations it is then asked to predict. Parks, Mehta
and Lester all fitted their fields to DFDR records, so predicting a DFDR record
partly re-derives the fit. Section 4's session-23 entry opens by saying so, and
section 7 has wanted a field that does not have that property since.

This is that field. `grads_file_202012301000.dat` is domain D03 of the LES
behind Yoshimura et al. 2023 (GRL 50, e2022GL101286): 2000 x 2000 x 80 on a
70 m Lambert grid over Honshu, 10:00 UTC 30 December 2020, produced by a weather
model and not by fitting an aeroplane. Flying it is the first load run in this
project whose wind never saw an accelerometer.

*** AND THE FIRST THING IT ESTABLISHES IS A LIMIT ON ITSELF. ***
The domain's top level is 7,900 m. `boeing747` cruises at 11,278 m, so THE
FIELD CANNOT SUPPORT A CRUISE-ALTITUDE COMPARISON AT ALL. Section 7 lists this
dataset as the route out of the circularity for the cruise CAT work; it is not,
because the event is not at cruise. What it can support is a comparison at ITS
altitude, and that is what this script does.

WHAT IS MATCHED, AND WHAT IS LEFT AS THE CONFOUND. Yoshimura flew 151 paths at
3,000 m, heading 090, at 148.8 m/s. This flies `boeing747` at the SAME altitude,
the SAME heading, along the SAME lines of latitude, at the SAME airspeed --
which the 747 trims at cleanly, alpha = 3.31 deg, well inside the linear range.
So airspeed, altitude, track and field are all common, and the ONE difference
left is the flight-dynamics code and the aircraft it carries. That is the same
design as `vortex_compare.py` against JSBSim, and it is why the result is a
comparison of SHAPE and not a validation of absolute load.

*** THE 747 IS FAR OUTSIDE ITS DECLARED ENVELOPE HERE, AND THE RUN SAYS SO. ***
Section 1's validation claim is a cruise claim -- a Mach and altitude band this
condition is nowhere near. `checks.recovery_band` is called and its verdict is
printed rather than suppressed. Nothing below is evidence for section 4; it is
evidence about the SHAPE of a response, which is the only thing a cross-code
comparison at a foreign condition can be.

HOW THE FIELD IS READ. GrADS binary, little-endian float32, one time step, the
variable order of `grads_file.ctl`: FLATsrf, FLONsrf, then Ustg, Vstg, Wstg, PT
and p at 80 levels each, then zssrf. That layout predicts a file of exactly
6,448,000,000 bytes and the file is exactly that, which is the first check that
the offsets below are right. Only a subvolume around the flight path is loaded.

THE GEOMETRY NEEDS NO PROJECTION CODE. FLATsrf and FLONsrf carry the latitude
and longitude of every native grid point, so the mapping is measured off the
file rather than reimplemented from the Lambert parameters. Measured at the
flight origin: +i is due east at 72.05 m per step and +j due north at 71.26 m,
with zero convergence -- the domain's central meridian is 140 E and the track
sits at 140.0998 E.

THE CHECK THAT MAKES ANY OF THIS TRUSTWORTHY. Yoshimura's own output records
the wind AT THE AIRCRAFT in three columns, along a track whose latitude and
longitude are also recorded. So this sampler is pointed at their path and asked
to reproduce their numbers before it is allowed to fly anything. That test
fixes the sign and axis conventions -- which of Ustg/Vstg/Wstg is east, north
and up, and which way their body axes point -- by measurement rather than by
assuming a convention and hoping.

Run: PYTHONPATH=<abs worktree root> .venv/Scripts/python.exe scripts/les_flight.py --outdir runs/cat
"""

import argparse
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

import atisim  # noqa: F401  -- enables x64
from atisim import airframe, checks, response, trim, validation, vortex_viz, wind
from atisim.aircraft import REGISTRY
from atisim.state import quat_to_dcm
from atisim.units import RAD2DEG

PALETTE = {"model": "#1D5D77", "reference": "#A9501C", "wind": "#3E6A48",
           "muted": "#7E8D93", "grid": "#D6DCD8"}

LES_DIR = Path("C:/Users/mateusz/UROP/yoshimura-figshare-21152203/les")
WORK = Path("C:/Users/mateusz/UROP/yoshimura-figshare-21152203/unpacked/"
            "flightsim-data/work")

# The four nested domains, and the flightsim output directory each one drove.
DOMAINS = {"D01": "500m", "D02": "250m", "D03": "70m", "D04": "35m"}

# Yoshimura's OWN ensemble through each domain, measured from their 151 output
# files by scripts/les_flight.py --summarise. Quoted here so a run has its
# counterpart beside it; recomputed rather than trusted when --summarise runs.
YOSH = {
    "500m": dict(rms=0.0310, sigma_w=2.250, peak=0.0400, peak_dn=0.207),
    "250m": dict(rms=0.0427, sigma_w=1.464, peak=0.0500, peak_dn=0.206),
    "70m":  dict(rms=0.0799, sigma_w=2.386, peak=0.0800, peak_dn=0.434),
    "35m":  dict(rms=0.1166, sigma_w=3.048, peak=0.1300, peak_dn=0.909),
}

class Ctl(NamedTuple):
    """Everything about one domain's binary, read from its own control file.

    The four domains are NOT the same grid -- D01 through D04 nest from 500 m
    down to 35 m and differ in extent as well as spacing -- so nothing here may
    be hardcoded from the one that happened to be read first.
    """

    nx: int
    ny: int
    nz: int
    dx: float                   # m, from pdef
    levels: np.ndarray          # (nz,) m
    var_order: tuple            # names, in file order
    var_levels: dict            # name -> number of levels
    dat: Path
    expected_bytes: int


def parse_ctl(ctl_path: Path) -> Ctl:
    """Read a GrADS control file into the layout of its binary.

    The size this predicts is checked against the file on disk before a byte is
    read, which is the cheapest possible test that the variable order and the
    level counts have been understood.
    """
    text = ctl_path.read_text().splitlines()
    nx = ny = nz = None
    dx = None
    levels: list[float] = []
    var_order: list[str] = []
    var_levels: dict[str, int] = {}
    mode = None
    for raw in text:
        line = raw.strip()
        low = line.lower()
        if low.startswith("pdef"):
            f = line.split()
            nx, ny = int(f[1]), int(f[2])
            dx = float(f[11])                       # LCCR: ... slon dx dy
            mode = None
        elif low.startswith("zdef"):
            f = line.split()
            nz = int(f[1])
            mode = "z" if "levels" in low else None
        elif low.startswith("vars"):
            mode = "v"
        elif low.startswith("endvars"):
            mode = None
        elif mode == "z":
            if any(c.isalpha() for c in line):
                mode = None
            else:
                levels.extend(float(v) for v in line.split())
        elif mode == "v" and line:
            f = line.split()
            n = int(f[1])
            var_order.append(f[0])
            var_levels[f[0]] = n if n > 0 else 1
    if None in (nx, ny, nz) or dx is None:
        raise SystemExit(f"{ctl_path} did not yield pdef/zdef")
    if len(levels) != nz:
        raise SystemExit(f"{ctl_path}: zdef says {nz} levels, {len(levels)} listed")
    dat = next(ctl_path.parent.glob("grads_file_*.dat"))
    total = sum(var_levels[v] for v in var_order)
    return Ctl(nx=nx, ny=ny, nz=nz, dx=dx, levels=np.array(levels),
               var_order=tuple(var_order), var_levels=var_levels, dat=dat,
               expected_bytes=total * nx * ny * 4)

# Yoshimura's own condition, from work/stdin and the flown output.
YOSH_LAT0, YOSH_LON0 = 35.25, 140.10
YOSH_ALT = 3000.0
# THEIR AIRSPEED, FROM THEIR SOURCE AND CONFIRMED BY THEIR OWN POSITION OUTPUT.
# `src/fs.f90` sets U_0 = 133 m/s, and `x_cor_path-*.txt` columns 3-4 run
# 79385.6 -> 92737.8 m in 100 s = 133.5 m/s with y constant to the metre.
#
# An earlier pass here took 148.8 m/s from their xlatlon lat/lon columns. That
# was wrong: `fs.f90` builds those columns as the inverse-distance-weighted
# AVERAGE of the grid latitude and longitude over the interpolation stencil,
# not as the aircraft's position, so they are smoothed and biased. The error
# was 12% in speed, which is 12% in encounter frequency and, since the gust
# load goes as w*V, about 12% in load.
YOSH_SPEED = 133.5
YOSH_HEADING_DEG = 90.0
AIRCRAFT = "boeing747"   # their derivative set is labelled B747 in fs.f90
RECORD_SECONDS = 100.0
SETTLE_SECONDS = 20.0


def _offset(c: Ctl, var: str) -> int:
    """Byte offset of a variable's first level, from the ctl variable order."""
    n = 0
    for name in c.var_order:
        if name == var:
            return n * c.nx * c.ny * 4
        n += c.var_levels[name]
    raise KeyError(var)


class LESField(NamedTuple):
    """A subvolume of the LES, and the affine map from flight NED into it."""

    u: Array          # (nk, nj, ni) eastward wind, m/s
    v: Array          # northward
    w: Array          # upward
    i0: float         # grid i of the flight origin, within the subvolume
    j0: float
    k0: float         # level index of the flight origin
    di_east: float    # m of true east per +i
    dj_north: float   # m of true north per +j
    cos_b: float      # flight +north axis, as a bearing in the LES
    sin_b: float
    dz: float         # m per level, checked uniform across the subvolume


def _catmull_rom(t: Array) -> tuple:
    """The four Catmull-Rom weights for a fractional offset in [0, 1].

    Interpolating -- w = (0, 1, 0, 0) at t = 0 and (0, 0, 1, 0) at t = 1, so
    the sampled field passes through the LES values rather than smoothing them
    -- and C1 continuous across cell boundaries, which is the whole point.
    """
    t2, t3 = t * t, t * t * t
    return (0.5 * (-t3 + 2.0 * t2 - t),
            0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
            0.5 * (-3.0 * t3 + 4.0 * t2 + t),
            0.5 * (t3 - t2))


def _tricubic(vol: Array, fk: Array, fj: Array, fi: Array) -> Array:
    """Tricubic (Catmull-Rom) sample of `vol` at fractional index.

    *** WHY NOT TRILINEAR, WHICH IS WHAT THIS USED TO BE. ***
    `wind.gust_rates` and `wind.gust_alphadot` take `jax.jacfwd` of the field,
    so the field's DERIVATIVE is a physical input and not just a convenience.
    Trilinear interpolation is C0: continuous in value, piecewise-constant in
    gradient. Differentiating it analytically therefore gives an `omega_gust`
    that STEPS at every cell boundary -- one step every 0.5 s at 70 m and
    133 m/s -- and the exactness of `jacfwd` is exactness about the wrong
    object. Yoshimura's own code avoids this from the other direction, by
    interpolating over a 216-point stencil and then finite-differencing a
    10-sample moving average; their gust rate is SMOOTHER than ours was,
    despite ours being "exact".

    Catmull-Rom is C1, so the gradient is continuous and `omega_gust` no longer
    steps. It reproduces a linear field EXACTLY -- its tangent estimate
    (p2 - p0)/2 is exact for linear data -- which is what lets the synthetic
    negative control in this file stay an equality test rather than becoming a
    tolerance.

    The stencil is 4x4x4 = 64 points against trilinear's 8. The base index is
    clamped so all four points exist on every axis.
    """
    nk, nj, ni = vol.shape
    fk = jnp.clip(fk, 1.0, nk - 2.001)
    fj = jnp.clip(fj, 1.0, nj - 2.001)
    fi = jnp.clip(fi, 1.0, ni - 2.001)
    k = jnp.floor(fk).astype(int) - 1
    j = jnp.floor(fj).astype(int) - 1
    i = jnp.floor(fi).astype(int) - 1
    wk = _catmull_rom(fk - jnp.floor(fk))
    wj = _catmull_rom(fj - jnp.floor(fj))
    wi = _catmull_rom(fi - jnp.floor(fi))
    out = 0.0
    for a in range(4):
        for b in range(4):
            for c in range(4):
                out = out + wk[a] * wj[b] * wi[c] * vol[k + a, j + b, i + c]
    return out


# The verification path samples the same way the flight does; one name so the
# two cannot drift apart.
_sample = _tricubic


def les_wind(pos_ned: Array, f: LESField) -> Array:
    """Wind in the flight NED frame at `pos_ned`, from the LES subvolume.

    The flight frame's +north axis is laid along an arbitrary LES bearing so
    that a run can follow Yoshimura's own easterly track while the harness
    still flies the aircraft along its own +north. Both the POSITION and the
    returned VECTOR are rotated, which is the pair that is easy to get half
    right: rotating only the position gives a field that is sampled correctly
    and points the wrong way.
    """
    north, east, down = pos_ned[0], pos_ned[1], pos_ned[2]
    true_north = north * f.cos_b - east * f.sin_b
    true_east = north * f.sin_b + east * f.cos_b
    fi = f.i0 + true_east / f.di_east
    fj = f.j0 + true_north / f.dj_north
    fk = f.k0 + (-down - YOSH_ALT) / f.dz
    u = _sample(f.u, fk, fj, fi)        # east
    v = _sample(f.v, fk, fj, fi)        # north
    w = _sample(f.w, fk, fj, fi)        # up
    n_comp = v * f.cos_b + u * f.sin_b
    e_comp = -v * f.sin_b + u * f.cos_b
    return jnp.array([n_comp, e_comp, -w])


def load_field(c: Ctl, *, lat0: float, lon0: float, alt: float,
               half_km: float, heading_deg: float) -> tuple[LESField, dict]:
    """Read the subvolume around a point, and measure the grid metric there."""
    dat = c.dat
    size = dat.stat().st_size
    if size != c.expected_bytes:
        raise SystemExit(f"{dat} is {size} bytes, the ctl layout predicts "
                         f"{c.expected_bytes} -- refusing to read it")
    NY, NX = c.ny, c.nx
    lat = np.memmap(dat, dtype="<f4", mode="r", offset=_offset(c, "FLATsrf"),
                    shape=(NY, NX))
    lon = np.memmap(dat, dtype="<f4", mode="r", offset=_offset(c, "FLONsrf"),
                    shape=(NY, NX))
    d = (np.asarray(lat) - lat0) ** 2 + (np.asarray(lon) - lon0) ** 2
    j0, i0 = np.unravel_index(int(np.argmin(d)), d.shape)

    R = 6371000.0
    m_lat = np.deg2rad(1.0) * R
    m_lon = np.deg2rad(1.0) * R * np.cos(np.deg2rad(lat0))
    di_east = float(lon[j0, i0 + 1] - lon[j0, i0]) * m_lon
    dj_north = float(lat[j0 + 1, i0] - lat[j0, i0]) * m_lat

    # ASYMMETRIC, because the flight is. Every path starts at the origin and
    # runs EAST, and successive paths step SOUTH; nothing ever goes far west or
    # north. A symmetric box wastes that, and on D04 -- 35 m spacing -- a
    # symmetric 52 km box is 1.05 GB of jit constant against 55 MB for the
    # bounds the run actually needs.
    east_lo, east_hi = -2000.0, 1000.0 * half_km
    north_lo, north_hi = -4000.0, 2000.0
    ia = max(i0 + int(east_lo / abs(di_east)) - 4, 0)
    ib = min(i0 + int(east_hi / abs(di_east)) + 4, NX - 1)
    ja = max(j0 + int(north_lo / abs(dj_north)) - 4, 0)
    jb = min(j0 + int(north_hi / abs(dj_north)) + 4, NY - 1)
    # The pad is in METRES, not levels. D04 is spaced at 35 m against 100 m for
    # the other three, so a fixed level count would give it +/- 140 m -- less
    # than the altitude the run actually drifts through, and `_sample` would
    # clamp instead of failing.
    k_mid = int(np.argmin(np.abs(c.levels - alt)))
    dz_here = float(np.diff(c.levels[max(k_mid - 1, 0):k_mid + 2]).mean())
    k_pad = int(round(600.0 / dz_here)) + 2
    ka, kb = max(k_mid - k_pad, 0), min(k_mid + k_pad, c.nz - 1)
    steps = np.diff(c.levels[ka:kb + 1])
    if np.ptp(steps) > 1e-6:
        raise SystemExit(f"levels are not uniform over {ka}..{kb}: {steps}")
    dz = float(steps[0])

    def _read(var):
        base = _offset(c, var)
        out = np.empty((kb - ka + 1, jb - ja + 1, ib - ia + 1), dtype=np.float32)
        for n, k in enumerate(range(ka, kb + 1)):
            lev = np.memmap(dat, dtype="<f4", mode="r",
                            offset=base + k * NX * NY * 4, shape=(NY, NX))
            out[n] = lev[ja:jb + 1, ia:ib + 1]
        return out

    u, v, w = _read("Ustg"), _read("Vstg"), _read("Wstg")

    # *** THE MEAN FLOW IS REMOVED, AND THAT IS A CHOICE WITH A REASON. ***
    # The subvolume carries about 26 m/s of mean easterly. Left in, it does two
    # things, one harmless and one not: it produces no load, because a uniform
    # wind is a Galilean shift the air-relative aero cannot see -- and it
    # carries the aeroplane over the ground 18% faster, so it meets the eddies
    # 18% sooner and every frequency in the response moves with it.
    #
    # Yoshimura removed it too, which is not an assumption here but a
    # measurement: `verify_against_yoshimura` regresses their recorded columns
    # on this field and finds intercepts of -24.9 and +7.7 m/s on u and v
    # against +0.08 on w. A zero-mean quantity shows no offset; the two with a
    # mean do. So their columns are perturbations, and matching them is what
    # makes the encounter rate comparable rather than 18% out.
    u_mean, v_mean = float(u.mean()), float(v.mean())
    u = u - u_mean
    v = v - v_mean
    for name, arr in (("Ustg", u), ("Vstg", v), ("Wstg", w)):
        if not np.isfinite(arr).all() or np.abs(arr).max() > 200.0:
            raise SystemExit(f"{name} subvolume looks wrong: "
                             f"max |value| {np.abs(arr).max():.1f} m/s")
    b = np.deg2rad(heading_deg)
    # float32, deliberately: the LES itself is float32, and the volume becomes a
    # compile-time constant inside the jitted scan -- at float64 that is 70 MB
    # of constant to trace through for no precision the source ever had.
    field = LESField(
        u=jnp.asarray(u, dtype=jnp.float32), v=jnp.asarray(v, dtype=jnp.float32),
        w=jnp.asarray(w, dtype=jnp.float32),
        i0=float(i0 - ia), j0=float(j0 - ja), k0=float(k_mid - ka),
        di_east=di_east, dj_north=dj_north,
        cos_b=float(np.cos(b)), sin_b=float(np.sin(b)), dz=dz,
    )
    meta = dict(i0=int(i0), j0=int(j0), di_east=di_east, dj_north=dj_north,
                shape=u.shape, ka=ka, kb=kb,
                sigma_w=float(w[k_mid - ka].std()),
                sigma_u=float(u[k_mid - ka].std()),
                u_mean=u_mean, v_mean=v_mean, dz=dz, k_mid=k_mid,
                level=float(c.levels[k_mid]),
                mb=u.nbytes * 3 / 1e6,
                lat=float(lat[j0, i0]), lon=float(lon[j0, i0]))
    return field, meta


def verify_against_yoshimura(field: LESField, meta: dict, path_file: Path,
                             cor_file: Path) -> dict:
    """Sample this reader along THEIR track and compare to THEIR wind columns.

    Their columns 13-15 are the wind at the aircraft in their body axes, and
    columns 16-17 are the latitude and longitude they were at. So the sampler
    can be asked the same question at the same places, and the answer either
    matches or the offsets, the axis order or the signs are wrong.
    """
    a = np.loadtxt(path_file, delimiter=",")
    lat, lon = a[:, 15], a[:, 16]
    uw, vw, ww = a[:, 12], a[:, 13], a[:, 14]
    alt = np.loadtxt(cor_file, delimiter=",")[:, 4]
    if len(alt) != len(lat):
        raise SystemExit(f"{path_file.name} has {len(lat)} rows and "
                         f"{cor_file.name} has {len(alt)} -- they must align "
                         "row for row or the altitudes are attached to the "
                         "wrong positions")
    take = slice(None, None, 10)
    lat, lon, uw, vw, ww, alt = (x[take] for x in (lat, lon, uw, vw, ww, alt))

    R = 6371000.0
    m_lat = np.deg2rad(1.0) * R
    m_lon = np.deg2rad(1.0) * R * np.cos(np.deg2rad(meta["lat"]))
    tn = (lat - meta["lat"]) * m_lat
    te = (lon - meta["lon"]) * m_lon
    fi = field.i0 + te / field.di_east
    fj = field.j0 + tn / field.dj_north
    fk = field.k0 + (alt - YOSH_ALT) / field.dz
    su = np.asarray(jax.vmap(_sample, (None, 0, 0, 0))(field.u, fk, fj, fi))
    sv = np.asarray(jax.vmap(_sample, (None, 0, 0, 0))(field.v, fk, fj, fi))
    sw = np.asarray(jax.vmap(_sample, (None, 0, 0, 0))(field.w, fk, fj, fi))

    def sample(vol, di, dj, dk):
        return np.asarray(jax.vmap(_sample, (None, 0, 0, 0))(
            vol, fk + dk, fj + dj, fi + di))

    out = {"n": len(lat), "components": {}}
    for name, vol, theirs in (("u", field.u, uw), ("v", field.v, vw),
                              ("w", field.w, ww)):
        raw = sample(vol, 0.0, 0.0, 0.0)
        # Which of THEIR columns does this component match best? The answer
        # must be the expected one, and by a clear margin -- that is the part
        # of this check that is allowed to gate the run, because it tests the
        # variable order and the sign convention rather than the registration.
        rs = {k: float(np.corrcoef(raw, v)[0, 1])
              for k, v in (("u", uw), ("v", vw), ("w", ww))}
        # A REGISTRATION residual, searched but NEVER applied. It is reported so
        # the reader knows how far off the alignment is; feeding it back into
        # the sampler would tune the field to reproduce the answer and turn the
        # only end-to-end check into a calibration.
        best = (0.0, 0.0, 0.0, rs[name])
        for dk_ in np.arange(0.0, 2.01, 0.25):
            for dj_ in np.arange(-1.5, 1.51, 0.25):
                for di_ in np.arange(-0.5, 2.01, 0.25):
                    r = float(np.corrcoef(sample(vol, di_, dj_, dk_), theirs)[0, 1])
                    if abs(r) > abs(best[3]):
                        best = (di_, dj_, dk_, r)
        q = sample(vol, best[0], best[1], best[2])
        slope, intercept = np.polyfit(q, theirs, 1)
        out["components"][name] = dict(
            r_raw=rs[name], matches=rs, best=best,
            slope=float(slope), intercept=float(intercept),
            our_sd=float(raw.std()), their_sd=float(theirs.std()))
    return out


def _highpass(x, dt, f_cut):
    """Drop everything below `f_cut`, by zeroing those bins and transforming back.

    NOT cosmetic. With fixed controls a 100 s record at 148.8 m/s holds about
    1.8 phugoid cycles -- the phugoid period here is 4.44 V/g = 67 s -- and it
    is lightly damped, so the low-frequency end of the raw record is an
    excursion of the flight condition rather than a response to turbulence.
    Yoshimura's aircraft HOLDS its altitude to 42 m, so their record carries no
    phugoid at all; comparing their rms against a raw rms of this one would be
    comparing a turbulence response against a turbulence response plus a
    phugoid. This is the term that makes the two comparable, and the drift it
    removes is reported separately rather than hidden by it.
    """
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), dt)
    X[f < f_cut] = 0.0
    return np.fft.irfft(X, n=len(x))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=Path("runs/cat"))
    ap.add_argument("--domain", default="D03", choices=sorted(DOMAINS))
    ap.add_argument("--flights", type=int, default=24)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--aircraft", default=AIRCRAFT,
                    help="registry entry to fly. boeing747 matches Yoshimura's "
                         "class; boeing737_approach is the entry whose data was "
                         "actually recovered near this altitude and speed.")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    res = DOMAINS[args.domain]
    ctl = parse_ctl(LES_DIR / args.domain / "grads_file.ctl")
    flightsim = WORK / f"results-1000_mand_{res}.nc-2D"

    print("=" * 78)
    print(f"ATISIM THROUGH YOSHIMURA'S LES FIELD -- {args.domain} ({res} grid)")
    print("=" * 78)
    print(f"  ctl     {ctl.nx} x {ctl.ny} x {ctl.nz}, pdef dx {ctl.dx:.0f} m, "
          f"levels {ctl.levels[0]:.0f}..{ctl.levels[-1]:.0f} m")
    print(f"  binary  {ctl.dat.name}, ctl predicts {ctl.expected_bytes:,} B")
    print(f"  vars    {', '.join(ctl.var_order)}")

    # The run covers 120 s at 148.8 m/s = 17.9 km of track, and successive
    # paths step 0.001 deg south, so the box has to hold both. `_sample`
    # CLAMPS at the edges rather than raising, so a box that is too small
    # would quietly flatten the field instead of failing -- hence the margin
    # and the track-extent assertion after the first flight.
    field, meta = load_field(ctl, lat0=YOSH_LAT0, lon0=YOSH_LON0,
                             alt=YOSH_ALT, half_km=24.0,
                             heading_deg=YOSH_HEADING_DEG)
    print(f"  origin  grid (i={meta['i0']}, j={meta['j0']}) = "
          f"{meta['lat']:.5f} N, {meta['lon']:.5f} E")
    print(f"  metric  +i = {meta['di_east']:.2f} m east, "
          f"+j = {meta['dj_north']:.2f} m north  (no convergence at 140 E)")
    print(f"  loaded  subvolume {meta['shape']} (k {meta['ka']}..{meta['kb']}), "
          f"{meta['mb']:.0f} MB")
    print(f"  at the flight level: sigma_w {meta['sigma_w']:.3f} m/s, "
          f"sigma_u {meta['sigma_u']:.3f} m/s")
    print(f"  mean flow REMOVED (declared): u {meta['u_mean']:+.2f}, "
          f"v {meta['v_mean']:+.2f} m/s -- see load_field")

    print()
    print("-" * 78)
    print("A. THE READER, CHECKED AGAINST YOSHIMURA'S OWN SAMPLED WIND")
    print("-" * 78)
    v = verify_against_yoshimura(field, meta,
                                 flightsim / "xlatlon_path-0001.txt",
                                 flightsim / "x_cor_path-0001.txt")
    print(f"  {v['n']} points along their path 1, every 10th step")
    print("  Their aircraft flies 090, so body x is east, body y SOUTH, body z")
    print("  down -- the expected signs are +1 on u and -1 on both v and w.")
    print()
    print("   ours   best match among their three columns        r as read")
    ok = True
    for name, c in v["components"].items():
        pick = max(c["matches"], key=lambda k: abs(c["matches"][k]))
        margin = abs(c["matches"][pick]) - max(
            abs(r) for k, r in c["matches"].items() if k != pick)
        flag = "OK" if (pick == name and margin > 0.15) else "*** MISMATCH ***"
        ok &= pick == name and margin > 0.15
        print(f"    {name}     -> their {pick}  (r={c['matches'][pick]:+.3f}, "
              f"margin {margin:+.3f})   {c['r_raw']:+.4f}   {flag}")
    print()
    print("  REGISTRATION, searched and reported but NOT applied:")
    for name, c in v["components"].items():
        di_, dj_, dk_, r = c["best"]
        print(f"    {name}: best r={r:+.4f} at di={di_:+.2f} dj={dj_:+.2f} "
              f"dk={dk_:+.2f}   their = {c['slope']:+.3f}*ours {c['intercept']:+.2f}")
    print(f"    -> a 1-2 cell horizontal and ~1.3 level vertical residual, i.e.")
    print(f"       under 150 m. It is NOT fed back into the sampler: fitting an")
    print(f"       offset until the check passes would turn the only end-to-end")
    print(f"       check into a calibration. The run below uses ZERO offset.")
    print()
    print("  AND ONE REAL FINDING ABOUT THEIR SETUP. The u and v intercepts are")
    print("  -24.9 and +7.7 m/s while w's is +0.08: their recorded wind has the")
    print("  MEAN FLOW REMOVED, which w -- being zero-mean already -- does not")
    print("  show. So their columns are gust perturbations, not total wind.")
    if not ok:
        print()
        print("  *** THE VARIABLE ORDER OR SIGN CONVENTION IS WRONG. Everything")
        print("      below would be a run through a misread field, so it is not")
        print("      run. ***")
        return

    print()
    print("-" * 78)
    print("B. THE CONDITION, AND WHY IT IS NOT SECTION 1's")
    print("-" * 78)
    ac = REGISTRY[args.aircraft]
    x, _ = trim.trim(jnp.array(YOSH_SPEED), jnp.array(YOSH_ALT), ac)
    print(f"  {args.aircraft} at {YOSH_ALT:.0f} m, {YOSH_SPEED:.1f} m/s, heading "
          f"{YOSH_HEADING_DEG:.0f} -- Yoshimura's own condition")
    print(f"  trim alpha {float(x[0]) * RAD2DEG:.2f} deg")
    from atisim.aircraft import CRUISE
    home = CRUISE[args.aircraft]
    print(f"  this entry's own recovery condition: {home['altitude']:.0f} m, "
          f"{home['airspeed']:.1f} m/s")
    print(f"  LES top level is {ctl.levels[-1]:.0f} m; boeing747 cruises at 11278 m.")
    print("  THE FIELD CANNOT REACH CRUISE, so this is a comparison at ITS")
    print("  altitude and NOT the cruise-CAT check section 7 wanted from it.")
    # `recovery_band` reads a flown trajectory, so it is reported in section C
    # once there is one -- printing a guess here would be worse than waiting.

    print()
    print("-" * 78)
    print("C. FLOWN")
    print("-" * 78)
    # ONE field closure and ONE wind model for every path. `integrate.rollout`
    # is jitted with `wind_model` STATIC, so a per-path closure would retrace
    # the whole scan and bake the volume in again as a fresh constant for each
    # flight. Paths are separated by their START POSITION instead, which is what
    # `fly_from_state` exists for -- it is the entry point vortex_compare.py
    # uses to start atisim from another engine's state.
    #
    # With heading 090 the flight frame's +east axis maps to true SOUTH, so
    # stepping the start east by 0.001 deg of latitude walks successive paths
    # south exactly as Yoshimura's own stdin does.
    field_fn = lambda p: les_wind(p, field)              # noqa: E731
    # `sampled_field_model`, not `field_model`. The tangent at the CG is the
    # worst estimator precisely here: an LES field has structure at the scale of
    # its own grid, and at 35 m that is a fifth of this aeroplane's span. The
    # sampled form fits a secant across the airframe instead, which is the
    # quantity the aerodynamics actually integrate over -- and it is the
    # better-motivated version of the 10-sample moving average Yoshimura's code
    # applies before differencing.
    #
    # *** AND IT IS GATED, BECAUSE FOR MOST OF THE REGISTRY IT CANNOT BE USED. ***
    # `airframe.stations` runs aft from the CG by `effective_tail_arm`, which is
    # -Cmq/CLq. Both 737 entries carry CLq = 0 EXACTLY, so that arm is infinite,
    # the longitudinal stations run to -inf, and `sampled_rates` samples the
    # field at infinity and returns NaN -- silently, all the way to a NaN rms.
    # `airframe.tail_arm_is_plausible` exists to catch precisely this and is
    # False for four of the seven registry entries (both 737s, and both light
    # aircraft, whose arms fall below the 2-chord band). It is not called at the
    # point of use anywhere else in the tree, including cat_bounds.py, which
    # gets away with it only by never flying anything but the 747.
    if airframe.tail_arm_is_plausible(ac):
        model = wind.sampled_field_model(field_fn, airframe.stations(ac))
        rate_estimator = "sampled_rates, secant across the airframe"
    else:
        model = wind.field_model(field_fn)
        rate_estimator = (f"gust_rates, tangent at the CG -- {args.aircraft} has "
                          f"an implausible tail arm "
                          f"({float(airframe.effective_tail_arm(ac)):.3g} chords) "
                          "so the sampled estimator is REFUSED, not silently NaN")
    print(f"  gust rates: {rate_estimator}")
    lat_step_m = 0.001 * np.deg2rad(1.0) * 6371000.0
    controls = trim.trimmed_controls(x[1], x[2])
    base = trim.trimmed_state(jnp.array(float(x[0])), jnp.array(YOSH_SPEED),
                              jnp.array(YOSH_ALT))
    records, drifts, first = [], [], None
    for n in range(args.flights):
        st = base._replace(pos_ned=jnp.array([0.0, n * lat_step_m, -YOSH_ALT]))
        dcm = quat_to_dcm(st.quat)
        st = st._replace(vel_body=st.vel_body + dcm.T @ field_fn(st.pos_ned))
        enc = vortex_viz.fly_from_state(
            ac, field_fn, st, controls, label=f"les-{n}",
            seconds=SETTLE_SECONDS + RECORD_SECONDS, dt=args.dt,
            window=(-1e12, 1e12), window_name="whole run", wind_model=model)
        if first is None:
            first = enc
            reach = float(np.asarray(enc.north).max())
            span_m = (field.u.shape[2] - 1 - field.i0) * abs(field.di_east)
            print(f"  track reaches {reach/1000:.1f} km along +east; the box holds "
                  f"{span_m/1000:.1f} km  {'OK' if reach < span_m else '*** TOO SMALL ***'}")
        k = int(round(SETTLE_SECONDS / args.dt))
        nz = np.asarray(enc.n_z)[k:]
        records.append(nz)
        speed = np.linalg.norm(np.asarray(enc.log.vel_body)[k:], axis=1)
        drifts.append(dict(alt=float(np.ptp(np.asarray(enc.altitude)[k:])),
                           spd=float(np.ptp(speed)),
                           alpha=float(np.ptp(np.asarray(enc.alpha_air)[k:]) * RAD2DEG)))
    hp = [_highpass(r, args.dt, response.PHUGOID_FLOOR_HZ) for r in records]
    rms = np.array([r.std() for r in records])
    rms_hp = np.array([r.std() for r in hp])
    spectra = [response.spectrum(r, args.dt) for r in hp]
    f_hz = spectra[0][0]
    psd = np.mean([s[1] for s in spectra], axis=0)
    peak = response.peak_frequency(f_hz, psd)
    modes = validation.longitudinal_modes(ac, float(x[0]), float(x[1]), float(x[2]),
                                          YOSH_SPEED, YOSH_ALT)
    f_sp = float(modes[-1][0]) / (2 * np.pi)

    print(f"  {args.flights} flights x {RECORD_SECONDS:.0f} s, first "
          f"{SETTLE_SECONDS:.0f} s discarded, dt {args.dt}")
    print(f"  n_z rms, raw     {rms.mean():.4f} g [{rms.min():.4f}, {rms.max():.4f}]")
    print(f"  n_z rms, phugoid removed above {response.PHUGOID_FLOOR_HZ} Hz")
    print(f"                   {rms_hp.mean():.4f} g [{rms_hp.min():.4f}, {rms_hp.max():.4f}]")
    print(f"  peak |dn|, raw   {max(np.abs(r).max() for r in records):.3f} g")
    print(f"  peak |dn|, h-p   {max(np.abs(r).max() for r in hp):.3f} g")
    print(f"  response peak    {peak:.4f} Hz   (spectrum of the high-passed record)")
    print(f"  short period     {f_sp:.4f} Hz  -> {100*(peak/f_sp-1):+.1f}%")
    worst = {k: max(d[k] for d in drifts) for k in drifts[0]}
    print(f"  condition drift, worst: altitude {worst['alt']:.0f} m, "
          f"speed {worst['spd']:.1f} m/s, |alpha| range {worst['alpha']:.2f} deg")
    band = checks.recovery_band(first.log, ac)
    print(f"  checks.recovery_band: passed={band.passed}")
    print(f"    {band.detail}")

    print()
    y = YOSH[res]
    print(f"  YOSHIMURA'S OWN {res} ENSEMBLE, same field, same condition:")
    print(f"    n_z rms {y['rms']:.4f} g, peak {y['peak']:.4f} Hz, "
          f"peak |dn| {y['peak_dn']:.3f} g, field sigma_w {y['sigma_w']:.3f} m/s")
    print(f"    ratio AtiSim/Yoshimura on rms: {rms_hp.mean()/y['rms']:.2f}x")
    print("    (their aircraft and their code -- the one difference left)")

    lv = np.arange(0.05, 1.21, 0.05)
    secs = sum(len(r) for r in hp) * args.dt
    print()
    print("  upcrossing rate, high-passed, per second:")
    for level in (0.1, 0.2, 0.3, 0.4, 0.6, 0.8):
        n = sum(((r[:-1] < level) & (r[1:] >= level)).sum() for r in hp)
        print(f"    {level:5.2f} g   {n/secs:8.4f}")


    np.save(args.outdir / f"les-nz-{args.domain}-{args.aircraft}.npy",
            np.array(records))
    with (args.outdir / "les-summary.csv").open("a") as fh:
        fh.write(f"{args.domain},{res},{ctl.dx:.0f},{args.aircraft},"
                 f"{meta['sigma_w']:.4f},{rms.mean():.4f},{rms_hp.mean():.4f},"
                 f"{max(np.abs(r).max() for r in hp):.4f},{peak:.4f},{f_sp:.4f},"
                 f"{y['rms']:.4f},{y['peak']:.4f},{band.passed}\n")
    fig_path = args.outdir / f"13-les-{args.domain}-{args.aircraft}.png"
    _figure(fig_path, f_hz, psd, hp, spectra, peak, f_sp, args.dt, args.aircraft)
    print(f"\n  figure -> {fig_path}")


def _figure(path, f_hz, psd, records, spectra, peak, f_sp, dt, aircraft):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8))
    t = np.arange(len(records[0])) * dt
    for r in records[:6]:
        ax.plot(t, r, lw=.7, color=PALETTE["model"], alpha=.55)
    ax.set_xlabel("time, s", fontsize=9)
    ax.set_ylabel("$n_z$, g", fontsize=9)
    ax.set_title(f"A. AtiSim's {aircraft} through the LES, first six paths"
                 " -- phugoid removed above 0.05 Hz", fontsize=10.5, loc="left")

    m = (f_hz > 0.02) & (f_hz < 3)
    bx.loglog(f_hz[m], psd[m], lw=1.6, color=PALETTE["model"],
              label="AtiSim, ensemble mean")
    bx.axvline(f_sp, color=PALETTE["reference"], ls="--", lw=1.2,
               label=f"short period {f_sp:.3f} Hz")
    bx.axvline(peak, color=PALETTE["wind"], ls=":", lw=1.6,
               label=f"response peak {peak:.3f} Hz")
    bx.set_xlabel("frequency, Hz", fontsize=9)
    bx.set_ylabel("$n_z$ PSD, g$^2$/Hz", fontsize=9)
    bx.set_title("B. and where the response energy sits", fontsize=10.5, loc="left")
    bx.legend(fontsize=8)
    for a in (ax, bx):
        a.grid(True, color=PALETTE["grid"], lw=.6, alpha=.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle(f"Yoshimura's LES field, flown by AtiSim ({aircraft}) -- 3,000 m, "
                 "148.8 m/s, heading 090. NOT a cruise condition.",
                 fontsize=10.5, x=.005, ha="left")
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

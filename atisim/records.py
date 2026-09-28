"""The published record of an encounter, to set beside a run of the same case.

A run of a sourced case can be read against what the source measured. This
module finds that record for a run and puts it on the run's time axis. Two
kinds of record exist here:

* A TRACE, a recorded time history. Only the Hannibal encounter has one:
  Parks, Wingrove, Bach & Mehta 1985 (J. Aircraft 22(2)) Fig. 6, the DC-10's
  normal acceleration and vertical wind, digitised by
  `scripts/digitise_parks_fig6_altitude.py` into
  `atisim/data/parks1985_fig6_altitude.csv`. Each column of ink is read as its
  top and bottom, so the trace is a band, not a line. Nothing is smoothed.
* A BAND, a published range with no time history. TM-102186 states the
  Hannibal load "from +1.7 to -1.0 g" (`wind.TM102186_HANNIBAL_NZ`). Wingrove &
  Bach 1994 Fig. 8 puts the load increment of its vortex, updraft and
  manoeuvre records between -2.01 and -1.69 g (`vortex_viz.FIG8_LOAD_BAND`),
  read as an increment from level flight, as the manoeuvre preset reads it.

THE ALIGNMENT IS DECLARED. The record's clock is GMT and a run's clock is its
own. The trace is moved in time so that the record's deepest downdraft falls
on the run's deepest downdraft. That aligns the INPUTS, the wind the aircraft
met, and leaves the loads free to differ. Aligning on the load itself would
make the loads agree by construction.

THE AIRCRAFT DIFFER. Every record here is a DC-10 (Parks p. 127 and p. 128),
and the model flies the 747. The comparison is of order and size, never of
exact values, as `run._ORDERING_CAVEAT` states for the Parks and Wingrove &
Bach cases.
"""

import csv
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from atisim.vortex_viz import FIG8_LOAD_BAND
from atisim.wind import TM102186_HANNIBAL_NZ

PARKS_FIG6 = Path(__file__).parent / "data" / "parks1985_fig6_altitude.csv"
PARKS_SOURCE = ("Parks, Wingrove, Bach & Mehta 1985, J. Aircraft 22(2) 124-129, "
                "Fig. 6, p. 127")
TM102186_SOURCE = "NASA TM-102186, pp. 3-4"
WINGROVE_FIG8_SOURCE = "Wingrove & Bach 1994, J. Aircraft 31(4) 753-760, Fig. 8"

ALIGNMENT = ("moved in time so that the record's deepest downdraft falls on the "
             "run's deepest downdraft (DECLARED)")


class Trace(NamedTuple):
    """A recorded time history on the run's clock, as a band of ink."""

    t: np.ndarray  # s, on the run's clock
    lo: np.ndarray  # g, the bottom of the ink
    hi: np.ndarray  # g, the top of the ink
    label: str
    source: str
    shift: float  # s, added to the record's clock: a run time is record time + shift


class Band(NamedTuple):
    """A published range of the load factor, with no time history."""

    lo: float  # g
    hi: float  # g
    label: str
    source: str


class Record(NamedTuple):
    """What the source measured, ready to draw on a run's load-factor axis."""

    case: str  # the encounter, in words
    trace: Trace | None
    bands: tuple[Band, ...]
    note: str  # one sentence on what the reader compares


@lru_cache(maxsize=1)
def parks_fig6() -> dict:
    """The digitised Fig. 6 panels: {(panel, curve): array of (t, value)}, by time.

    `t` is seconds after 01:21 GMT. The load is in g and the vertical wind in
    ft/s, up positive, as the figure prints them.
    """
    out: dict = {}
    with open(PARKS_FIG6, newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault((row["panel"], row["curve"]), []).append(
                (float(row["t_s_after_0121"]), float(row["value"])))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def _deepest_downdraft_s() -> float:
    """The record's time of its deepest downdraft, from the bottom of the ink."""
    w = parks_fig6()[("vertical_wind", "ink_bottom")]
    return float(w[np.argmin(w[:, 1]), 0])


def hannibal_trace(t_run: np.ndarray, w_up_run: np.ndarray) -> Trace | None:
    """The DC-10's load through Hannibal, on the run's clock, for the run's span.

    `w_up_run` is the vertical wind the run met, up positive. None when the run
    met no downdraft to align on, or when the aligned record does not reach the
    run's span.
    """
    t_run, w_up_run = np.asarray(t_run, float), np.asarray(w_up_run, float)
    if t_run.size < 2 or not np.isfinite(w_up_run).any() or np.nanmin(w_up_run) >= 0.0:
        return None
    shift = float(t_run[np.nanargmin(w_up_run)]) - _deepest_downdraft_s()
    d = parks_fig6()
    lo, hi = d[("normal_acceleration", "ink_bottom")], d[("normal_acceleration", "ink_top")]
    t = lo[:, 0] + shift
    inside = (t >= t_run[0]) & (t <= t_run[-1])
    if inside.sum() < 2:
        return None
    # Both envelopes are read at the same columns, so the times match row by row.
    top = np.interp(lo[:, 0], hi[:, 0], hi[:, 1])
    return Trace(t=t[inside], lo=lo[inside, 1], hi=top[inside],
                 label="DC-10 record (Parks 1985 Fig. 6)", source=PARKS_SOURCE,
                 shift=shift)


def _fig8_band(category: str) -> Band:
    lo, hi = FIG8_LOAD_BAND
    return Band(lo=1.0 + lo, hi=1.0 + hi,
                label=f"Wingrove & Bach Fig. 8: the lowest load of the DC-10 records, "
                      f"{1.0 + lo:+.2f} to {1.0 + hi:+.2f} g",
                source=f"{WINGROVE_FIG8_SOURCE}, the {category} category, read as an "
                       "increment from +1 g")


def category(kind: str, case: str | None = None) -> str | None:
    """The record a run of this field is read against: "hannibal", "vortex",
    "updraft", "manoeuvre", or None.

    `kind` is the spec's wind kind when the run has a spec (so the manoeuvre is
    told from still air), else `meta["wind_field"]["kind"]`. `case` is
    `meta["wind_field"]["case"]`. Mehta's five-vortex field is the Hannibal
    encounter identified again, so it is read against the Hannibal record.
    """
    if kind in ("VortexArray", "SingleVortex") and case in ("hannibal", "mehta"):
        return "hannibal"
    if kind == "MehtaHannibal":
        return "hannibal"
    if kind in ("VortexArray", "SingleVortex"):
        return "vortex"
    if kind == "UpdraftColumn":
        return "updraft"
    if kind == "manoeuvre":
        return "manoeuvre"
    return None


def for_run(kind: str, case: str | None, t_run, w_up_run) -> Record | None:
    """The record for a run of this field, or None for a field with no record."""
    which = category(kind, case)
    if which is None:
        return None
    if which == "hannibal":
        lo, hi = TM102186_HANNIBAL_NZ
        measured = Band(lo=lo, hi=hi, label=f"TM-102186: measured {hi:+.1f} to {lo:+.1f} g",
                        source=TM102186_SOURCE)
        trace = hannibal_trace(t_run, w_up_run)
        return Record(
            case="Hannibal, 3 April 1981",
            trace=trace, bands=(measured,),
            note=("The band is the DC-10's recorded load, " + ALIGNMENT + ". "
                  if trace is not None else "")
                 + "The DC-10 is not the 747, so compare the order and size of the "
                   "loads, not their exact values.")
    names = {"vortex": "vortex", "updraft": "updraft", "manoeuvre": "manoeuvring"}
    return Record(
        case=f"Wingrove & Bach, {names[which]} records",
        trace=None, bands=(_fig8_band(names[which]),),
        note="No time history of this encounter is held. The band is the range the "
             "paper gives for the lowest load of its DC-10 records. Compare the "
             "run's lowest load with it, not its shape.")


def as_dict(record: Record | None, max_points: int = 1500) -> dict | None:
    """The record as plain numbers, for the browser."""
    if record is None:
        return None
    trace = None
    if record.trace is not None:
        step = max(1, -(-record.trace.t.size // max_points))
        trace = {"t": record.trace.t[::step].round(3).tolist(),
                 "lo": record.trace.lo[::step].round(4).tolist(),
                 "hi": record.trace.hi[::step].round(4).tolist(),
                 "label": record.trace.label, "source": record.trace.source}
    return {"case": record.case, "trace": trace, "note": record.note,
            "bands": [b._asdict() for b in record.bands]}


def w_up_from_ned(wind_ned) -> np.ndarray:
    """The vertical wind, up positive, from a NED wind history."""
    return -np.asarray(wind_ned, float)[:, 2]

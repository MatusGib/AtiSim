# Scripts

Run the command-line tools in `scripts/` from the repository root. Each one takes `--help`, and
each one's docstring says what it measures and which source it compares against. A script
given `--png` or `--outdir` writes its figures there; otherwise it opens a window.

## Flying and encounters

| Script | What it does |
|---|---|
| `fly.py` | Fly interactively with a cockpit display. `--aircraft`, `--wind {none,hannibal,morton,updraft}`, `--autopilot` to start engaged, and `--save run.npz` to keep the trajectory. |
| `vortex.py` | Fly a vortex encounter, an updraft and an elevator manoeuvre, and draw the analysis figure, including the Wingrove & Bach Fig. 8 discriminator. `--case {hannibal,morton}`, `--strip` for strip-integrated loads, and `--artifacts DIR` to save runs for the analysis app. |
| `microburst.py` | Penetrate an Oseguera–Bowles microburst on fixed controls, and measure the F-factor against the aircraft's thrust. The Cherokee is the default aircraft. |
| `leewave.py` | Fly the 747 through a Doyle et al. mountain lee wave, and ask whether its engines can counter it. |
| `lateral.py` | Fly vortex lines that vary across the span, and report the rolling response. |
| `analyse.py` | Re-analyse `.npz` runs saved by `fly.py` without flying them again. |
| `tune.py` | Draw step responses for tuning the autopilot gains. |

The analysis app reads the artifacts that `vortex.py --artifacts` writes:

```bash
python scripts/vortex.py --artifacts runs/analysis
python -m atisim.apps.sweep runs/analysis
```

## Validation

These scripts reproduce the evidence summarised in {doc}`validation`.

| Script | What it does |
|---|---|
| `sanity.py` | Eleven hand-derived checks, from degenerate inputs upward. |
| `checkpoint.py` | Trim the 747, hold it in level flight, and check its linear modes against CR-2144. |
| `cat_validation.py` | The clear-air-turbulence cases: the Hannibal encounter against the recorded load, the TM-102186 Fig. 8 ordering and its mechanism, a third CR-2144 flight condition, and the Misaka severity index. `--outdir`. |
| `cat_ensemble.py` | The three Fig. 8 categories with error bars, over a Dryden ensemble. `--seeds`, `--outdir`. |
| `cat_spectra.py` | Response spectra and load-exceedance rates over Dryden ensembles. `--seeds`, `--outdir`. |
| `vortex_compare.py` | Fly AtiSim through the vortex that JSBSim flew, and report where the two engines part. |
| `cr2144_digitisation_crosscheck.py` | Compare the two independent readings of CR-2144 pp. 220–222 that ship in `atisim/data/`. |

## Reference data

These scripts regenerate data that is already in the repository. The test suite reads the
stored copies, so you only need them to rebuild the data.

| Script | What it does | Needs |
|---|---|---|
| `gen_jsbsim_reference.py` | Freeze JSBSim's 737 trim, modes and responses into `atisim/tests/data/`. | `.[ref]` |
| `gen_jsbsim_vortex_reference.py` | Freeze JSBSim's response to the Wingrove & Bach vortex cases. | `.[ref]` |
| `gen_jsbsim_747.py` | Recover the `boeing747_jsbsim` entry from JSBSim's own B747. | `.[ref]` |
| `digitise_mil_f_8785c_fig7.py` | Digitise MIL-F-8785C Figure 7, turbulence exceedance probability. | `--pdf`, a copy of MIL-F-8785C |

# NASA CR-2144 Boeing 747 aerodynamic data — digitised figures

Source: `Digitise_plots.pdf`, printed pages 218-228 (PDF pages 223-233 of the full
CR-2144 report). 25 panels, 78 curves, 77 emitted.

## Layout

```
csv/               one file per curve, named p<page>_<coefficient>_<altitude>.csv
overlays/          one PNG per printed page: digitised points drawn on the scan
verification.pdf   one page per panel: trace on the scan (left), replot from the CSV (right)
manifest.csv       every curve with its Mach span and point count
```

## Method

1. The scans are pulled out of the PDF as their embedded 300 dpi greyscale images
   (2544x3300), so nothing is resampled.
2. Axis lines are found morphologically; tick marks are found as short runs of ink
   touching the axis. Each panel is calibrated on its own printed ticks by a
   RANSAC fit, anchored on the axis line itself (which carries a known value).
   The p.219 panels are fitted in log10 space on both axes.
3. Flight-condition markers are found by Hough transform. Each marker is erased and
   the curve bridged straight through it, so the trace is not deflected by the ring
   or the numeral.
4. Curves are traced column by column from each marker outwards, with slope
   continuity and gap bridging for dashed and dash-dot styles.
5. Altitude is assigned from the markers whose Mach is unique to one curve
   (0.45 -> SL, 0.50 -> 20000 ft, 0.70 and 0.90 -> 40000 ft). The two ambiguous
   Machs (0.65 and 0.80) are resolved by which traced curve passes through them.
6. Traces are median-then-mean filtered over 9 columns and sampled at 0.005 Mach.

## Accuracy

Tick-fit residuals are 0.2-3.5 px across all 50 axes, i.e. better than 0.004 in Mach
and better than 0.5% of full scale in the ordinate. That is the calibration error only.
The dominant error is the printed line width (~9 px, roughly 1.5% of a typical full
scale) plus scan distortion, so treat individual readings as good to about 1-2% of
full scale, not better.

Marker readings are listed in each file header as a cross-check. On p.220 C_Lalpha
the 40000 ft curve reads 5.003 / 4.883 / 5.541 at FC8 / FC9 / FC10.

The `[model carries: not supplied]` field in each header is a placeholder: the
reference values from your model were not available in this session. Supply the
FC3-FC10 table and it can be populated.

## Known gaps and interpretive calls

* p.222 C_DM has no sea-level curve: at SL it is drawn along the zero axis and is
  not separable from it. No file is emitted rather than a fabricated row of zeros.
* Where two curves cross and swap branches (notably p.220 C_Lalpha near M 0.93), the
  data has a short gap at the crossing. The two branches either side were recovered
  with a manually placed trace seed; the assignment of branch to altitude there
  follows the printed line style, not an automatic test.
* Panels marked `poor` below have all three altitude curves printed within about a
  line width of each other over much of the Mach range. The three files are still
  emitted, but the split between them is partly interpretive and the curves should
  be treated as one band with a spread of roughly the line width.
* Every manual intervention (clip window, forced marker, extra seed) is recorded in
  the header of the file it affected, and the full override table is in
  `overrides_applied.txt`.

## Per-panel confidence

| page | panel | curves | Mach span | confidence | note |
|---|---|---|---|---|---|
| 218 | alpha0 | 3/3 | 0.29-0.93 | good | clean separation; solid curve continues below the axis past M 0.66 as printed |
| 218 | delta_s | 3/3 | 0.30-0.93 | fair | all three curves sit within ~2 deg; markers 4/6 and 7/9 forced by value |
| 219 | CL | 3/3 | 0.22-1.00 | good | log-log; straight lines, well separated |
| 219 | CD | 3/3 | 0.29-0.92 | fair | crowded right-hand end, markers 9 and 10 nearly touch |
| 220 | CL_alpha | 3/3 | 0.30-0.99 | good | 20k/40k branches after the M 0.93 crossing recovered by extra seeds; small gap at the crossing |
| 220 | CD_alpha | 3/3 | 0.29-0.98 | good | steep upper dashed branch recovered by extra seed |
| 221 | Cm_alpha | 3/3 | 0.29-0.93 | good | 40k plunge to FC10 partially traced |
| 221 | Cm_adot | 3/3 | 0.29-0.93 | poor | three curves overlap heavily between M 0.45 and 0.75; altitude split here is partly interpretive |
| 221 | Cm_q | 3/3 | 0.33-0.97 | good | clean separation |
| 222 | CL_M | 3/3 | 0.35-0.93 | poor | curves printed on top of each other below M 0.8; treat the three files as one band |
| 222 | CD_M | 2/3 | 0.67-0.93 | fair | SL not drawn (lies on the zero axis) - no file emitted |
| 222 | Cm_M | 3/3 | 0.29-0.96 | good | clean apart from the deep 20k trough near M 0.87 |
| 223 | CL_de | 3/3 | 0.29-0.95 | good |  |
| 223 | Cm_de | 3/3 | 0.30-0.96 | good |  |
| 224 | Cy_beta | 3/3 | 0.29-0.96 | poor | all three within ~0.05 rad^-1; split is partly interpretive |
| 224 | Cn_beta | 3/3 | 0.30-0.95 | fair | SL and 20k coincide left of M 0.5 |
| 224 | Cl_beta | 3/3 | 0.30-0.93 | good | 40k trough at M 0.82 recovered by extra seed |
| 225 | Cl_p | 3/3 | 0.29-0.96 | poor | all three within ~0.01 rad^-1; split is partly interpretive |
| 225 | Cn_p | 3/3 | 0.29-0.92 | fair | steep near-vertical branches at M 0.42 and M 0.88 only partly traced |
| 226 | Cl_r | 3/3 | 0.30-0.93 | good |  |
| 226 | Cn_r | 3/3 | 0.29-0.96 | good |  |
| 227 | Cl_da | 3/3 | 0.30-0.96 | fair | SL and 20k cross near M 0.48; SL mid-section recovered by extra seed |
| 227 | Cn_da | 3/3 | 0.29-0.94 | fair | curves converge near the zero crossing at M 0.8 |
| 228 | Cy_dr | 3/3 | 0.31-0.95 | good |  |
| 228 | Cn_dr | 3/3 | 0.28-0.95 | good |  |
| 228 | Cl_dr | 3/3 | 0.29-0.94 | fair | all three low-amplitude and close together |


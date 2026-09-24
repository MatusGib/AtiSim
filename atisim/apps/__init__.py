"""Dash front ends over `atisim.analysis`.

**This is the only package allowed to import Dash, and it computes nothing.**
Every number it displays comes from `atisim.checks` or
`atisim.analysis.series`, each asserted by a test: add the computation to a
module, assert it in a test, then display it.

The app also never runs the simulator. `n_steps` is a `static_argname`, so every
distinct dt pays a fresh JAX compile, and a panel whose contents depend on how
warm the machine is is not a check. Runs are flown by the scripts and read from
disk.
"""

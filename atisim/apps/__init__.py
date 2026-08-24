"""Dash front ends over `atisim.analysis`.

**This is the only package allowed to import Dash, and it computes nothing.**
Every number it displays comes from `atisim.checks` or
`atisim.analysis.series`, each asserted by a test. That is the protocol
`docs/ASSUMPTIONS.md` states for the notebook -- add the computation to a module,
assert it in a test, then display it -- and session 13 exists because session 12
did only the middle step and left the notebook telling a reader something the
code had already disproved. A figure is more persuasive than a print, so the same
drift here would be worse.

The app also never runs the simulator. `n_steps` is a `static_argname`, so every
distinct dt pays a fresh 0.6-0.9 s JAX compile, and a panel whose contents depend
on how warm the machine is is not a check. Runs are flown by the scripts and read
from disk.
"""

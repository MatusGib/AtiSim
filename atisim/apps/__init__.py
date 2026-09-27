"""Dash front ends over `atisim.run` and `atisim.analysis`.

**This is the only package allowed to import Dash, and it computes nothing.**
Every number it displays comes from `atisim.run`, `atisim.checks` or
`atisim.analysis`, each asserted by a test: add the computation to a module,
assert it in a test, then display it.

Runs are flown through `atisim.run.fly`, in one background thread
(`jobs.py`), never inside a callback: `n_steps` is a `static_argname`, so every
distinct dt pays a fresh JAX compile, and a panel whose contents depend on how
warm the machine is is not a check. Results reads what a run wrote to disk.

    shell.py    header, workspace switch, status bar, router
    start.py    Start: presets and recent runs
    setup.py    Setup: the Model Builder (tree, settings, preview, dock)
    results.py  Results: the analysis deep dive (moved from sweep.py)
    sweep.py    `python -m atisim.apps.sweep DIR`: the app, opened on Results
    theme.py    palette, Mantine theme and the Plotly template, in one place
    components.py  icons, status words, provenance markers
    jobs.py     the single worker thread and its job registry
"""

"""The analysis layer: run artifacts, and the figures drawn from them.

Deliberately a separate subpackage from the simulator, and it imports one way
only -- `flightsim.analysis` may import `flightsim`, never the reverse. That is
the same separation `viz.py` already states for the trajectory log: *"the log has
to be readable with no simulator in sight"*.

Needs the `ui` extra (`pyarrow`, `plotly`). Nothing in `flightsim/` proper
imports this package, so the simulator and every script in `scripts/` keep
working without it.
"""

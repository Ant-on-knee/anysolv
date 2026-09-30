"""Stub for fairchem.core.graph.compute.generate_graph.
"""

from __future__ import annotations


def generate_graph(*_args, **_kwargs):  # pragma: no cover
    raise RuntimeError(
        "generate_graph() called, but anysolv builds the radius graph externally "
        "(otf_graph must be False). Edges should be supplied via data.py."
    )

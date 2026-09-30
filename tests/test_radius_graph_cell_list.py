"""Validate radius_graph_cell_list against the brute-force radius_graph: identical edge SET.

The cell list is the O(N) non-PBC fast path; this asserts it returns the EXACT same directed
edge set as the O(N^2) reference across sizes, densities, and degenerate cell layouts, then
shows the timing gap (and a size where the brute-force cdist can't even allocate).

The reference is called with an effectively infinite max_neighbors so its degeneracy cap never
fires - the cell list has no cap, so this is the apples-to-apples comparison (the cap is a
no-op at molecular densities anyway; see radius_graph).

    python anysolv/tests/test_radius_graph_cell_list.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from anysolv.data import radius_graph, radius_graph_cell_list  # noqa: E402

NO_CAP = 10**9  # disable radius_graph's max-neighbor cap for the comparison


BOUNDARY_TOL = 1e-3  # two correct impls may disagree only for pairs this close to the cutoff


def edge_key(ei: torch.Tensor, n: int) -> torch.Tensor:
    """Encode a directed edge set as int64 keys (src,dst with src/dst < n)."""
    return ei[0].long() * n + ei[1].long()


def compare(a: torch.Tensor, b: torch.Tensor, pos: torch.Tensor, cutoff: float, n: int):
    """(match, n_boundary_ties, max_dev): edge sets must agree except for pairs whose TRUE
    distance is within BOUNDARY_TOL of `cutoff` - the only place a sqrt-vs-matmul distance
    (cell list vs cdist) can legitimately split a tie."""
    ka, kb = edge_key(a, n), edge_key(b, n)
    diff = torch.cat([ka[~torch.isin(ka, kb)], kb[~torch.isin(kb, ka)]])
    if diff.numel() == 0:
        return True, 0, 0.0
    dist = (pos[diff // n] - pos[diff % n]).norm(dim=1)
    dev = (dist - cutoff).abs()
    return bool((dev <= BOUNDARY_TOL).all()), int(diff.numel()), float(dev.max())


def cloud(n: int, density: float = 0.05) -> torch.Tensor:
    """n atoms uniformly in a cube sized for `density` atoms/angstrom^3 (organic liquids ~0.03-0.1)."""
    box = (n / density) ** (1 / 3)
    return torch.rand(n, 3) * box


def main() -> int:
    torch.manual_seed(0)
    cutoff = 6.0
    cases = [
        ("empty", torch.zeros(1, 3)),                                # n < 2 edge case
        ("tiny", cloud(5)),
        ("small", cloud(200)),
        ("dense-cluster", torch.randn(800, 3) * 3.0),                # very high coordination
        ("flat-slab", cloud(1500) * torch.tensor([1.0, 1.0, 0.04])), # one near-zero axis
        ("medium", cloud(3000)),
        ("large", cloud(8000)),
    ]

    ok = True
    print(f"{'case':16s} {'n':>6} {'edges':>10} {'brute(s)':>9} {'cell(s)':>9} {'ties':>5}  match")
    for name, pos in cases:
        n = pos.shape[0]
        t0 = time.perf_counter(); eb = radius_graph(pos, cutoff, NO_CAP); tb = time.perf_counter() - t0
        t0 = time.perf_counter(); ec = radius_graph_cell_list(pos, cutoff); tc = time.perf_counter() - t0
        m, n_ties, dev = compare(eb, ec, pos, cutoff, max(n, 1))
        ok &= m
        tag = "OK" if m else f"MISMATCH (max dev {dev:.1e})"
        print(f"{name:16s} {n:6d} {eb.shape[1]:10d} {tb:9.3f} {tc:9.3f} {n_ties:5d}  {tag}")

    # A size where the reference's [N,N] cdist is ~9 GB - cell list only.
    n = 30000
    pos = cloud(n)
    t0 = time.perf_counter(); ec = radius_graph_cell_list(pos, cutoff); tc = time.perf_counter() - t0
    print(f"{'xlarge(cell)':16s} {n:6d} {ec.shape[1]:10d} {'-':>9} {tc:9.3f} {'-':>5}  (brute: O(N^2) mem)")

    print("\nALL EDGE SETS MATCH" if ok else "\n*** MISMATCH PRESENT ***")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

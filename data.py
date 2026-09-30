"""
Build the input tensor dict + non-PBC radius graph.
"""

from __future__ import annotations

import torch


class AtomicData(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    def get(self, key, default=None): 
        return self[key] if key in self else default


def radius_graph(pos: torch.Tensor, cutoff: float, max_neighbors: int,
                 degeneracy_tol: float = 0.01) -> torch.Tensor:
    """Non-PBC radius graph matching fairchem's non-strict max-neighbors rule.
    Returns edge_index [2, E] object where each col is a pair of neighbors
    """
    n = pos.shape[0]
    if n < 2:
        return torch.zeros(2, 0, dtype=torch.long, device=pos.device)
    d = torch.cdist(pos, pos)
    within = (d <= cutoff) & ~torch.eye(n, dtype=torch.bool, device=pos.device)
    src, dst = [], []
    for i in range(n):
        nbr = torch.nonzero(within[i], as_tuple=False).flatten()
        if nbr.numel() == 0:
            continue
        if nbr.numel() > max_neighbors:
            dd = d[i, nbr]
            order = torch.argsort(dd)
            nbr, dd = nbr[order], dd[order]
            eff = dd[max_neighbors] + degeneracy_tol
            nbr = nbr[dd <= eff]
        src.append(nbr)
        dst.append(torch.full_like(nbr, i))
    if not src:
        return torch.zeros(2, 0, dtype=torch.long, device=pos.device)
    return torch.stack([torch.cat(src), torch.cat(dst)])


def radius_graph_cell_list(pos: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Linear scaling veersion of radius_graph
    """
    n = pos.shape[0]
    device = pos.device
    if n < 2:
        return torch.zeros(2, 0, dtype=torch.long, device=device)

    # 1. bin into cells of side == cutoff, origin at the bounding-box min corner
    cell_xyz = torch.floor((pos - pos.min(dim=0).values) / cutoff).long()      # [N,3] >= 0
    dims = cell_xyz.max(dim=0).values + 1                                       # cells per axis
    strides = torch.tensor([1, int(dims[0]), int(dims[0]) * int(dims[1])], device=device)
    cid = (cell_xyz * strides).sum(dim=1)                                       # [N] flat cell id
    num_cells = int(dims.prod())

    # 2. dense cell -> atom table; capacity == max occupancy so nothing is dropped
    counts = torch.bincount(cid, minlength=num_cells)
    cap = int(counts.max())
    order = torch.argsort(cid)
    offsets = torch.cumsum(counts, 0) - counts                                  # first slot/cell
    rank = torch.arange(n, device=device) - offsets[cid[order]]                 # within-cell rank
    table = torch.full((num_cells, cap), -1, dtype=torch.long, device=device)
    table[cid[order], rank] = order

    # 3. candidates = atoms in the 27 cells around each atom's cell
    shifts = torch.tensor(
        [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
        device=device,
    )                                                                           # [27,3]
    nbr_xyz = cell_xyz[:, None, :] + shifts                                      # [N,27,3]
    in_box = ((nbr_xyz >= 0) & (nbr_xyz < dims)).all(dim=2)                      # [N,27]
    nbr_cid = (nbr_xyz * strides).sum(dim=2)                                     # [N,27]
    nbr_cid = torch.where(in_box, nbr_cid, torch.zeros_like(nbr_cid))           # dummy-safe index
    cand = table[nbr_cid].reshape(n, 27 * cap)                                  # [N,27*cap]
    cand = torch.where(
        in_box[:, :, None].expand(n, 27, cap).reshape(n, 27 * cap),
        cand, torch.full_like(cand, -1),                                        # mask oob cells
    )

    # 4. distance filter - mirror radius_graph: sqrt distance, <= cutoff, drop self
    centers = torch.arange(n, device=device)[:, None].expand_as(cand)
    real = cand >= 0
    dist = (pos[:, None, :] - pos[cand.clamp(min=0)]).norm(dim=2)                # [N,27*cap]
    keep = real & (cand != centers) & (dist <= cutoff)
    return torch.stack([cand[keep], centers[keep]])


# Above this many atoms, build_atomic_data switches from the O(N^2) brute-force radius_graph
# to the O(N) cell list (validated edge-set-identical in tests/test_radius_graph_cell_list.py).
_CELL_LIST_MIN_ATOMS = 2000


def _as_arrays(atoms_or_arrays):
    """Accept an ase.Atoms or a (numbers, positions) pair; return (Z[np], R[np])."""
    import numpy as np

    if hasattr(atoms_or_arrays, "get_positions"):  # ase.Atoms
        atoms = atoms_or_arrays
        return (np.asarray(atoms.get_atomic_numbers()),
                np.asarray(atoms.get_positions(), dtype=float))
    numbers, positions = atoms_or_arrays
    return np.asarray(numbers), np.asarray(positions, dtype=float)


def build_atomic_data(
    atoms_or_arrays,
    charge: int = 0,
    spin: int = 1,
    solvent: torch.Tensor | None = None,
    cutoff: float = 6.0,
    max_neighbors: int = 300,
    dataset: str = "omol",
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> AtomicData:
    """Assemble the backbone input for a single molecule.

    `solvent` is an optional pre-normalized (1, 8) tensor (see anysolv.solvent); pass None
    for the plain (gas/no-solvent) checkpoints.
    """
    numbers, positions = _as_arrays(atoms_or_arrays)
    n = len(numbers)

    pos = torch.tensor(positions, dtype=dtype, device=device)
    atomic_numbers = torch.tensor(numbers, dtype=torch.long, device=device)
    batch = torch.zeros(n, dtype=torch.long, device=device)
    natoms = torch.tensor([n], dtype=torch.long, device=device)

    if n > _CELL_LIST_MIN_ATOMS:
        edge_index = radius_graph_cell_list(pos, cutoff).to(device)
    else:
        edge_index = radius_graph(pos, cutoff, max_neighbors).to(device)
    n_edges = edge_index.shape[1]
    cell = torch.zeros(1, 3, 3, dtype=dtype, device=device)
    cell_offsets = torch.zeros(n_edges, 3, dtype=dtype, device=device)
    nedges = torch.tensor([n_edges], dtype=torch.long, device=device)

    data = AtomicData(
        pos=pos,
        atomic_numbers=atomic_numbers,
        batch=batch,
        natoms=natoms,
        charge=torch.tensor([int(charge)], dtype=torch.long, device=device),
        spin=torch.tensor([int(spin)], dtype=torch.long, device=device),
        dataset=[dataset],
        edge_index=edge_index,
        cell=cell,
        cell_offsets=cell_offsets,
        nedges=nedges,
    )
    if solvent is not None:
        data["solvent"] = solvent.to(dtype=dtype, device=device)
    return data

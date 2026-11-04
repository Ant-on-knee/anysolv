"""Public inference API: predict_solvation_energy -> (dE in eV, dF in eV/angstrom).

dE = E_solv - E_gas 
Auto-selects the best available model (model_moe > model_compact)
"""

from __future__ import annotations

import numpy as np
import torch

from ._backbone._compat.inference import InferenceSettings, guess_inference_settings
from .data import build_atomic_data
from .model import default_checkpoint, load_model
from .solvent import get_solvent_vector

_MODEL_CACHE: dict = {}

_DEFAULT_SOLVENT = object()

def _get_model(checkpoint, device, dtype, inference_settings):
    settings = guess_inference_settings(inference_settings)
    key = (str(checkpoint), str(device), str(dtype), repr(settings))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(checkpoint, device=device, dtype=dtype,
                                       inference_settings=settings)
    return _MODEL_CACHE[key]

def _unwrap(x, key):
    return x[key] if isinstance(x, dict) else x


def _prepare_prediction(
    atoms_or_arrays,
    charge,
    spin,
    solvent,
    checkpoint,
    device,
    dtype,
    inference_settings,
):
    if checkpoint is None:
        checkpoint = default_checkpoint()
    model = _get_model(checkpoint, device, dtype, inference_settings)

    if solvent is _DEFAULT_SOLVENT:
        solvent = "water"
    solvent_vec = get_solvent_vector(
        solvent, eps_transform=getattr(model, "eps_transform", None)
    )
    data = build_atomic_data(
        atoms_or_arrays,
        charge=charge,
        spin=spin,
        solvent=solvent_vec,
        dtype=dtype,
        device=device,
    )
    return model, data


def predict_solvation_energy(
    atoms_or_arrays,
    charge: int = 0,
    spin: int = 1,
    solvent=_DEFAULT_SOLVENT,
    checkpoint: str | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    inference_settings: str | InferenceSettings = "default",
):
    """Return (dEsolv (eV): float, dFsolv (eV/A): np.ndarray[n_atoms, 3]).

    atoms_or_arrays   : an ase.Atoms, or a (atomic_numbers, positions[angstrom]) tuple.
    charge, spin      : total charge and spin multiplicity (defaults 0 / 1).
    solvent           : solvent name (str), or None for the gas/vacuum baseline. 
                        default solvent is water
    checkpoint        : None (default; auto-selects 'model_moe' if its weights are present, else 'model_compact'), 
                        a checkpoint name, or a path to a valid .pt.
    dtype             : torch.float32 (default) or torch.float64.
    inference_settings: 'default' (reference implementation), 
                        'fast' (block-GEMM SO2 + tf32 + torch.compile, merge-free), 
                        'fast_gpu' (adds Triton kernels; CUDA-only),
                        or a custom InferenceSettings. 
                        Recommended: use 'fast' for compact models and 'fast_gpu' for MoLE models
    """
    model, data = _prepare_prediction(
        atoms_or_arrays,
        charge,
        spin,
        solvent,
        checkpoint,
        device,
        dtype,
        inference_settings,
    )

    rmsd = model.norm["energy"]["rmsd"]
    mean = model.norm["energy"]["mean"]

    out = model(data)
    raw_e = _unwrap(out["energy"], "energy").reshape(-1)[0]
    raw_f = _unwrap(out["forces"], "forces")  # [n_atoms, 3], conservative (autograd) forces

    # unnormalize energies and forces
    delta_e = float((raw_e * rmsd + mean).item())
    delta_f = (raw_f * rmsd).detach().cpu().numpy().astype(np.float64)
    return delta_e, delta_f


def predict_solvation_hessian(
    atoms_or_arrays,
    charge: int = 0,
    spin: int = 1,
    solvent=_DEFAULT_SOLVENT,
    checkpoint: str | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    inference_settings: str | InferenceSettings = "default",
    *,
    vmap: bool = False,
) -> np.ndarray:
    """Return the analytical AnySolv correction Hessian in eV/angstrom^2.

    The result has shape ``(3N, 3N)`` in atom-major Cartesian order. The
    low-memory loop implementation is the default; ``vmap=True`` is faster for
    small systems but can require substantially more memory.

    Analytical Hessians require the ``general`` execution backend. With the
    default inference settings this function reuses the same process-cached
    model as :func:`predict_solvation_energy`; no checkpoint reload occurs.
    Float64 is recommended for accurate second derivatives.
    """
    model, data = _prepare_prediction(
        atoms_or_arrays,
        charge,
        spin,
        solvent,
        checkpoint,
        device,
        dtype,
        inference_settings,
    )
    output = model.predict_hessian(data, vmap=vmap)
    raw_hessian = _unwrap(output["hessian"], "hessian").squeeze(0)
    rmsd = model.norm["energy"]["rmsd"]
    return (raw_hessian * rmsd).detach().cpu().numpy().astype(np.float64)

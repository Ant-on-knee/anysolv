"""anysolv - package which adds a standalone solvation correction for a given geometry and solvent

Loads a converted anysolv checkpoint and returns dE = E_solv - E_gas plus the
matching force correction, as an additive term for any gas-phase potential. Depends only
on PyTorch + numpy (ASE optional, for I/O). See README.md.

    from anysolv import predict_solvation_energy
    dE, dF = predict_solvation_energy((atomic_numbers, positions), charge=0, spin=1)
"""

from ._backbone._compat.inference import InferenceSettings
from .model import default_checkpoint_path, load_model, print_default_checkpoint_path
from .predict import predict_solvation_energy

__all__ = [
    "predict_solvation_energy",
    "load_model",
    "default_checkpoint_path",
    "print_default_checkpoint_path",
    "InferenceSettings",
]

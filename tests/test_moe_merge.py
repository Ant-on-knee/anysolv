"""Unit test for the MoE MOLE-merge fast path.

Merging the MOLE experts (settings.merge_mole=True) collapses the per-system expert mixture into a
plain eSCNMDBackbone with single Linear SO2 layers.

Checks the following:
  * merge_MOLE_model must pass `solvent` to csd_embedding (solvent-conditioned model),
  * the merged backbone must be moved to the source device/dtype.

Run from the repo root:  python -m pytest anysolv/tests/test_moe_merge.py -q
"""

from __future__ import annotations

import torch

from anysolv._backbone._compat.inference import InferenceSettings
from anysolv._backbone.escn_md import MLP_EFS_Head, eSCNMDBackbone
from anysolv._backbone.escn_moe import eSCNMDMoeBackbone
from anysolv.data import build_atomic_data
from anysolv.solvent import get_solvent_vector

_DTYPE = torch.float64
_NUMBERS = [8, 1, 1]
_POS = [[0.0, 0.0, 0.0], [0.0, 0.757, 0.587], [0.0, -0.757, 0.587]]


def _tiny_moe():
    torch.manual_seed(0)
    backbone = eSCNMDMoeBackbone(
        num_experts=2,
        max_num_elements=100,
        sphere_channels=8,
        lmax=2,
        mmax=2,
        otf_graph=False,
        edge_channels=5,
        num_distance_basis=7,
        use_dataset_embedding=False,
        use_solvent_embedding=True,
        solvent_emb_hidden=8,
        solvent_output_gate=True,
        always_use_pbc=False,
        use_pbc=False,
        use_quaternion_wigner=False,
        execution_mode="general",
        activation_checkpointing=False,
        direct_forces=False,
        regress_stress=False,
    )
    head = MLP_EFS_Head(backbone)
    return backbone.to(_DTYPE).eval(), head.to(_DTYPE).eval()


def _data():
    vec = get_solvent_vector("water", strict=False)
    d = build_atomic_data((_NUMBERS, _POS), charge=0, spin=1, solvent=vec, dtype=_DTYPE)
    d["pos"].requires_grad_(True)
    return d


def _ef(backbone, head):
    d = _data()  # one shared data object: forces differentiate the SAME pos the backbone consumed
    out = head(d, backbone(d))
    e = out["energy"]["energy"] if isinstance(out["energy"], dict) else out["energy"]
    f = out["forces"]["forces"] if isinstance(out["forces"], dict) else out["forces"]
    return e, f


def test_merge_matches_unmerged():
    """Merging the MOLE experts reproduces the un-merged energy/forces to float64 precision."""
    backbone, head = _tiny_moe()
    e0, f0 = _ef(backbone, head)

    merged = backbone.prepare_for_inference(_data(), InferenceSettings(merge_mole=True))
    # The merge returns a NEW plain (non-MoE) backbone -> block-GEMM / compile / Triton become safe.
    assert isinstance(merged, eSCNMDBackbone)
    assert not isinstance(merged, eSCNMDMoeBackbone)

    e1, f1 = _ef(merged, head)
    assert (e0 - e1).abs().max().item() < 1e-7
    assert (f0 - f1).abs().max().item() < 1e-7


def test_merged_composition_lock():
    """A merged model raises a clear error if called on a different composition (charge here)."""
    backbone, _ = _tiny_moe()
    d0 = _data()
    merged = backbone.prepare_for_inference(d0, InferenceSettings(merge_mole=True))
    merged.on_predict_check(d0)  # same composition -> no raise

    vec = get_solvent_vector("water", strict=False)
    d1 = build_atomic_data((_NUMBERS, _POS), charge=1, spin=1, solvent=vec, dtype=_DTYPE)
    try:
        merged.on_predict_check(d1)
    except (AssertionError, RuntimeError, ValueError):
        return
    raise AssertionError("expected on_predict_check to reject a different composition")


def test_merged_solvent_lock():
    """A merged model raises if called on a different solvent.

    The solvent vector is baked into the merge (csd_embedding) but is not part of the element
    composition, so the lock must check it explicitly -- otherwise a merged model would silently
    return wrong-solvent numbers when reused across solvents.
    """
    backbone, _ = _tiny_moe()
    d0 = _data()  # merged on water (see _data)
    merged = backbone.prepare_for_inference(d0, InferenceSettings(merge_mole=True))
    merged.on_predict_check(d0)  # same solvent -> no raise

    other = get_solvent_vector("acetonitrile", strict=False)
    d1 = build_atomic_data((_NUMBERS, _POS), charge=0, spin=1, solvent=other, dtype=_DTYPE)
    try:
        merged.on_predict_check(d1)
    except (AssertionError, RuntimeError, ValueError):
        return
    raise AssertionError("expected on_predict_check to reject a different solvent")

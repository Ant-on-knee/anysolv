"""Torch-only unit test for the solvent-mask output gate (no checkpoint required).

Run from the repo root:

    python -m pytest anysolv/tests/test_solvent_output_gate.py -q
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from anysolv._backbone.escn_md import MLP_EFS_Head, eSCNMDBackbone
from anysolv.data import build_atomic_data
from anysolv.solvent import _SOLVENT_STATS, SOLVENT_DESCRIPTOR_ORDER, get_solvent_vector

_DTYPE = torch.float64
# H2O-ish geometry; exact coordinates are irrelevant to the gate guarantee.
_NUMBERS = [8, 1, 1]
_POS = [[0.0, 0.0, 0.0], [0.0, 0.757, 0.587], [0.0, -0.757, 0.587]]


def _tiny_gated_model(solvent_output_gate: bool):
    torch.manual_seed(0)
    backbone = eSCNMDBackbone(
        max_num_elements=100,
        sphere_channels=4,
        lmax=2,
        mmax=2,
        otf_graph=False,
        edge_channels=5,
        num_distance_basis=7,
        use_dataset_embedding=False,
        use_solvent_embedding=True,
        solvent_emb_hidden=8,
        solvent_output_gate=solvent_output_gate,
        always_use_pbc=False,
        use_pbc=False,
        use_quaternion_wigner=False,
        execution_mode="general",
        direct_forces=False,
        regress_stress=False,
    )
    head = MLP_EFS_Head(backbone)
    return torch.nn.ModuleDict({"b": backbone, "h": head}).to(_DTYPE).eval()


def _run(model, solvent):
    vec = get_solvent_vector(solvent, strict=False)
    data = build_atomic_data(
        (_NUMBERS, _POS), charge=0, spin=1, solvent=vec, dtype=_DTYPE,
    )
    data["pos"].requires_grad_(True)
    out = model["h"](data, model["b"](data))
    e = out["energy"]["energy"] if isinstance(out["energy"], dict) else out["energy"]
    f = out["forces"]["forces"] if isinstance(out["forces"], dict) else out["forces"]
    return e, f


def test_gate_vacuum_is_exactly_zero():
    model = _tiny_gated_model(solvent_output_gate=True)
    e_vac, f_vac = _run(model, None)
    assert e_vac.abs().max().item() == 0.0
    assert f_vac.abs().max().item() == 0.0

    e_wat, f_wat = _run(model, "water")
    assert e_wat.abs().max().item() > 0.0
    assert f_wat.abs().max().item() > 0.0


def test_gate_off_is_nonzero_in_vacuum():
    model = _tiny_gated_model(solvent_output_gate=False)
    e_vac, _ = _run(model, None)
    assert e_vac.abs().max().item() > 0.0


def test_umas_fast_pytorch_matches_general():
    """The block-GEMM (umas_fast_pytorch) backend is an exact reorder of the general backend's
    weights, so on a non-MoE backbone it must reproduce energy/forces to float64 precision."""
    from anysolv._backbone._compat.inference import InferenceSettings

    def _build(execution_mode):
        # Same seed + construction order -> identical weights across the two backends.
        torch.manual_seed(0)
        backbone = eSCNMDBackbone(
            max_num_elements=100,
            sphere_channels=4,
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
            execution_mode=execution_mode,
            activation_checkpointing=False,
            direct_forces=False,
            regress_stress=False,
        )
        head = MLP_EFS_Head(backbone)
        return backbone.to(_DTYPE).eval(), head.to(_DTYPE).eval()

    vec = get_solvent_vector("water", strict=False)

    def _eval(backbone, head, settings=None):
        data = build_atomic_data(
            (_NUMBERS, _POS), charge=0, spin=1, solvent=vec, dtype=_DTYPE,
        )
        data["pos"].requires_grad_(True)
        if settings is not None:
            backbone = backbone.prepare_for_inference(data, settings)  # may return new obj
        out = head(data, backbone(data))
        e = out["energy"]["energy"] if isinstance(out["energy"], dict) else out["energy"]
        f = out["forces"]["forces"] if isinstance(out["forces"], dict) else out["forces"]
        return e, f

    bg, hg = _build("general")
    e0, f0 = _eval(bg, hg)

    bf, hf = _build("umas_fast_pytorch")
    e1, f1 = _eval(bf, hf, settings=InferenceSettings(execution_mode="umas_fast_pytorch"))

    assert (e0 - e1).abs().max().item() < 1e-7
    assert (f0 - f1).abs().max().item() < 1e-7


def test_encoding_is_vacuum_anchored():
    """Physical gas phase (n=1, eps=1, rest 0) normalizes to all zeros."""
    from anysolv.solvent import normalize

    vacuum_raw = [
        1.0 if name in ("n", "epsilon") else 0.0 for name in SOLVENT_DESCRIPTOR_ORDER
    ]
    assert normalize(vacuum_raw) == [0.0] * len(SOLVENT_DESCRIPTOR_ORDER)


def test_water_spot_values():
    import json

    json_path = Path(__file__).resolve().parents[1] / "_const" / "solvent_descriptors.json"
    water = json.loads(json_path.read_text())["solvents"]["water"]
    vec = get_solvent_vector("water")[0]
    i_n = SOLVENT_DESCRIPTOR_ORDER.index("n")
    i_eps = SOLVENT_DESCRIPTOR_ORDER.index("epsilon")
    i_gamma = SOLVENT_DESCRIPTOR_ORDER.index("gamma")
    # get_solvent_vector returns float32, so compare with a float32-scale tolerance.
    assert abs(
        vec[i_n].item() - (water["n"] - 1.0) / _SOLVENT_STATS["n"]["scale"]
    ) < 1e-5
    assert abs(
        vec[i_eps].item() - (1.0 - 1.0 / water["epsilon"]) / _SOLVENT_STATS["epsilon"]["scale"]
    ) < 1e-5
    assert abs(
        vec[i_gamma].item() - water["gamma"] / _SOLVENT_STATS["gamma"]["scale"]
    ) < 1e-5

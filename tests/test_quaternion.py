"""Tests for the quaternion rotation backend."""

from __future__ import annotations

import torch

from anysolv import InferenceSettings
from anysolv._backbone.common.quaternion import edge_to_wigner


def test_quaternion_wigner_is_the_default():
    assert InferenceSettings().use_quaternion_wigner


def test_quaternion_wigner_supports_double_backward():
    edges = torch.tensor(
        [
            [0.7, 0.2, -0.4],
            [-0.3, -0.8, 0.5],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    wigner, inverse = edge_to_wigner(edges, lmax=2)

    first = torch.autograd.grad(
        (wigner.square().sum() + inverse.sin().sum()),
        edges,
        create_graph=True,
    )[0]
    second = torch.autograd.grad(first.square().sum(), edges)[0]

    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    identity = torch.eye(9, dtype=torch.float64).expand(len(edges), -1, -1)
    torch.testing.assert_close(wigner @ inverse, identity, atol=1e-12, rtol=1e-12)

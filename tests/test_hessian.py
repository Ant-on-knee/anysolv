"""Tests for analytical second derivatives."""

from __future__ import annotations

import numpy as np
import torch

from anysolv import predict_solvation_energy, predict_solvation_hessian
from anysolv._backbone.outputs import compute_hessian_loop, compute_hessian_vmap


def test_hessian_loop_uses_output_component_as_row():
    coordinates = torch.tensor(
        [[0.2, -0.1], [0.4, 0.7]], dtype=torch.float64, requires_grad=True
    )
    flat = coordinates.flatten()
    # Deliberately nonconservative force field so a transpose is observable.
    forces = torch.stack(
        (
            flat[0] + 2.0 * flat[1],
            3.0 * flat[0] - flat[1],
            flat[2] + 4.0 * flat[3],
            -2.0 * flat[2] + flat[3],
        )
    )
    expected = torch.tensor(
        [
            [-1.0, -2.0, 0.0, 0.0],
            [-3.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -4.0],
            [0.0, 0.0, 2.0, -1.0],
        ],
        dtype=torch.float64,
    )

    loop = compute_hessian_loop(forces, coordinates, create_graph=False)
    torch.testing.assert_close(loop, expected)

    # Rebuild the graph because the loop releases it on its final component.
    coordinates = coordinates.detach().requires_grad_(True)
    flat = coordinates.flatten()
    forces = torch.stack(
        (
            flat[0] + 2.0 * flat[1],
            3.0 * flat[0] - flat[1],
            flat[2] + 4.0 * flat[3],
            -2.0 * flat[2] + flat[3],
        )
    )
    vectorized = compute_hessian_vmap(forces, coordinates, create_graph=False)
    torch.testing.assert_close(vectorized, expected)


def test_public_hessian_matches_force_finite_difference():
    numbers = np.array([8, 1, 1])
    positions = np.array(
        [
            [0.0, 0.0, 0.119262],
            [0.0, 0.763239, -0.477047],
            [0.0, -0.763239, -0.477047],
        ]
    )
    analytical = predict_solvation_hessian(
        (numbers, positions),
        checkpoint="model_compact",
        dtype=torch.float64,
        vmap=True,
    )

    step = 1e-4
    numerical = np.empty_like(analytical)
    for column in range(positions.size):
        plus = positions.copy().reshape(-1)
        minus = positions.copy().reshape(-1)
        plus[column] += step
        minus[column] -= step
        plus_force = predict_solvation_energy(
            (numbers, plus.reshape(-1, 3)),
            checkpoint="model_compact",
            dtype=torch.float64,
        )[1]
        minus_force = predict_solvation_energy(
            (numbers, minus.reshape(-1, 3)),
            checkpoint="model_compact",
            dtype=torch.float64,
        )[1]
        numerical[:, column] = -(
            plus_force.reshape(-1) - minus_force.reshape(-1)
        ) / (2.0 * step)

    np.testing.assert_allclose(analytical, numerical, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(analytical, analytical.T, rtol=0.0, atol=1e-12)

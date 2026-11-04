"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Quaternion Wigner rotations for the l <= 2 AnySolv backbone.
"""

from __future__ import annotations

import torch


_BLEND_START = -0.9
_BLEND_WIDTH = 1.8


def _smooth_step(t: torch.Tensor) -> torch.Tensor:
    """Return the C-infinity transition used by FAIRChem's two charts."""
    value = t.clamp(0.0, 1.0)
    eps = torch.finfo(value.dtype).eps
    denominator = (value * (1.0 - value)).clamp(min=eps)
    result = torch.sigmoid((2.0 * value - 1.0) / denominator)
    result = torch.where(value < eps, torch.zeros_like(result), result)
    return torch.where(value > 1.0 - eps, torch.ones_like(result), result)


def _edge_to_y_quaternion(edge: torch.Tensor) -> torch.Tensor:
    """Return FAIRChem's stable two-chart edge-to-Y quaternion."""
    ex, ey, ez = edge.unbind(dim=-1)
    chart1 = torch.stack((1.0 + ey, -ez, torch.zeros_like(ex), ex), dim=-1)
    chart2 = torch.stack((-ez, 1.0 - ey, ex, torch.zeros_like(ex)), dim=-1)
    eps = torch.finfo(edge.dtype).eps
    chart1 = chart1 / chart1.square().sum(dim=-1, keepdim=True).clamp(min=eps).sqrt()
    chart2 = chart2 / chart2.square().sum(dim=-1, keepdim=True).clamp(min=eps).sqrt()

    # q and -q describe the same rotation. Align their signs so normalized
    # linear interpolation follows the shorter path and cannot cancel.
    dot = (chart2 * chart1).sum(dim=-1, keepdim=True)
    chart2 = torch.where(dot < 0.0, -chart2, chart2)
    weight = _smooth_step((ey - _BLEND_START) / _BLEND_WIDTH).unsqueeze(-1)
    return torch.nn.functional.normalize(
        (1.0 - weight) * chart2 + weight * chart1,
        dim=-1,
    )


def _rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert ``(w, x, y, z)`` quaternions to Cartesian rotation matrices."""
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def _l2_rotation(rotation: torch.Tensor) -> torch.Tensor:
    """Lift Cartesian rotations into AnySolv's five-dimensional l=2 basis."""
    sqrt2 = 2.0**0.5
    sqrt6 = 6.0**0.5
    tensor_basis = rotation.new_tensor(
        (
            ((0, 1 / sqrt2, 0), (1 / sqrt2, 0, 0), (0, 0, 0)),
            ((0, 0, 0), (0, 0, 1 / sqrt2), (0, 1 / sqrt2, 0)),
            ((-1 / sqrt6, 0, 0), (0, -1 / sqrt6, 0), (0, 0, 2 / sqrt6)),
            ((0, 0, 1 / sqrt2), (0, 0, 0), (1 / sqrt2, 0, 0)),
            ((1 / sqrt2, 0, 0), (0, -1 / sqrt2, 0), (0, 0, 0)),
        )
    )
    rotated = torch.einsum(
        "nij,bjk,nlk->nbil", rotation, tensor_basis, rotation
    )
    representation = torch.einsum("aij,nbij->nab", tensor_basis, rotated)

    sqrt3_over_2 = 3.0**0.5 / 2.0
    transform = rotation.new_tensor(
        (
            (0, 0, 0, 1, 0),
            (1, 0, 0, 0, 0),
            (0, 0, -0.5, 0, -sqrt3_over_2),
            (0, 1, 0, 0, 0),
            (0, 0, sqrt3_over_2, 0, -0.5),
        )
    )
    return transform @ representation @ transform.T


def edge_to_wigner(
    edge_vectors: torch.Tensor, lmax: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return edge-to-Y Wigner matrices and their inverses."""
    if lmax > 2:
        raise ValueError("The standalone quaternion path supports lmax <= 2.")
    normalized = torch.nn.functional.normalize(edge_vectors, dim=-1)
    rotation = _rotation_matrix(_edge_to_y_quaternion(normalized))
    size = (lmax + 1) ** 2
    wigner = edge_vectors.new_zeros((len(edge_vectors), size, size))
    wigner[:, 0, 0] = 1.0
    if lmax >= 1:
        wigner[:, 1:4, 1:4] = rotation
    if lmax >= 2:
        wigner[:, 4:9, 4:9] = _l2_rotation(rotation)
    return wigner, wigner.transpose(1, 2).contiguous()

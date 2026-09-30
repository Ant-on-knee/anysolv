"""Assemble the backbone + head from a converted checkpoint and load weights.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn

from ._backbone._compat.inference import InferenceSettings, guess_inference_settings
from ._backbone.escn_md import MLP_EFS_Head, eSCNMDBackbone
from ._backbone.escn_moe import eSCNMDMoeBackbone

_CKPT_DIR = Path(__file__).resolve().parent / "models"

# "general" is the cpu reference path; 
# "umas_fast_pytorch" is the block-diagonal SO2 GEMM path (composition-independent on non-MoE checkpoints);
# "umas_fast_gpu" adds the vTriton Wigner-permute kernels 
# On a MoE checkpoint the GPU/block-GEMM paths require a MoE merge first (merge_mole=True, fixed composition)
_SUPPORTED_EXECUTION_MODES = {"general", "umas_fast_pytorch", "umas_fast_gpu"}


@contextmanager
def tf32_context_manager():
    """Enable TF32 matmuls for the duration of the block, restoring prior state on exit.

    TF32 trades a little float32 mantissa precision for speed on NVIDIA GPUs.
    """
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_prec = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
        torch.set_float32_matmul_precision(old_prec)

# Supported checkpoints, in auto-selection priority order. 
# 'model_moe' (highest accuracy, download from HF); 
# 'model_compact' (trained from scratch, ~25 MB) is included in the repo and always available.
# ('model1'/'model1_compact' and 'model_smd'/'model_smd_compact' are deprecated: superseded by these two.)
_CHECKPOINT_PRIORITY = ("model_moe", "model_compact")

def default_checkpoint() -> str:
    """Checkpoint used when the caller names none: the first of 'model_moe' > 
    'model_compact' whose .pt is present in the models dir."""
    for name in _CHECKPOINT_PRIORITY:
        if (_CKPT_DIR / f"{name}.pt").exists():
            return name
    return _CHECKPOINT_PRIORITY[-1]


def default_checkpoint_path() -> Path:
    """Filesystem path to the .pt that `load_model(checkpoint=None)` would load."""
    return _resolve(default_checkpoint())


def print_default_checkpoint_path() -> None:
    """Print the path to the default model checkpoint."""
    print(default_checkpoint_path())

_BACKBONES = {
    "eSCNMDBackbone": eSCNMDBackbone,        # non-MoE, solvent-conditioned (model_compact)
    "eSCNMDMoeBackbone": eSCNMDMoeBackbone,  # UMA-S-1.2 mixture-of-experts (model_moe)
}
_DEFAULT_BACKBONE = "eSCNMDMoeBackbone"

# Inference overrides applied on top of the checkpoint's backbone_config. 
_INFERENCE_OVERRIDES = dict(
    otf_graph=False,            # edges precomputed by anysolv.data
    use_pbc=False,
    use_pbc_single=False,
    always_use_pbc=False,
    use_quaternion_wigner=False,  # Euler/Jd path
    activation_checkpointing=False,
    regress_forces=True,
    direct_forces=False,        # conservative forces via autograd
    regress_stress=False,
    regress_hessian=False,
)


class AnySolvModel(nn.Module):
    """backbone + EFS head; forward(data) -> head output dict (raw, un-normalized).
    """

    def __init__(self, backbone: nn.Module, head: nn.Module, norm: dict,
                 settings: InferenceSettings | None = None):
        super().__init__()
        self.backbone = backbone
        self.output_heads = nn.ModuleDict({"efs": head})
        self.norm = norm  # {energy:{mean,rmsd}, forces:{mean,rmsd}}
        self._settings = settings or InferenceSettings()
        self._prepared = False
        self._run = None  # eager or torch.compile'd _raw_forward, set on first forward

    def _raw_forward(self, data) -> dict:
        emb = self.backbone(data)
        return self.output_heads["efs"](data, emb)

    def _prepare(self, data) -> None:
        # prepare_for_inference may RETURN A NEW backbone (the MoE-merge path), so reassign.
        self.backbone = self.backbone.prepare_for_inference(data, self._settings)
        self._run = self._raw_forward
        if self._settings.compile:
            try:
                torch._dynamo.config.recompile_limit = 32
                self._run = torch.compile(self._raw_forward, dynamic=True)
            except Exception as exc:  # pragma: no cover - environment dependent
                logging.warning("torch.compile failed (%s); running eager", exc)
                self._run = self._raw_forward
        self._prepared = True

    def forward(self, data) -> dict:
        if not self._prepared:
            self._prepare(data)
        else:
            self.backbone.on_predict_check(data)
        ctx = tf32_context_manager() if self._settings.tf32 else nullcontext()
        with ctx:
            return self._run(data)


def _resolve(checkpoint: str | Path) -> Path:
    """Path to a checkpoint: an existing path, or a name looked up as models/<name>.pt."""
    p = Path(checkpoint)
    if p.exists():
        return p
    cand = _CKPT_DIR / f"{checkpoint}.pt"
    if cand.exists():
        return cand
    raise FileNotFoundError(
        f"checkpoint {checkpoint!r} not recognized (looked for {p} and {cand}). "
        f"Supported checkpoints: {_CHECKPOINT_PRIORITY}, or a path to a converted .pt."
    )


def load_model(checkpoint: str | Path | None = None, device: str = "cpu",
               dtype: torch.dtype = torch.float32,
               inference_settings: str | InferenceSettings = "default") -> AnySolvModel:
    """Build and load the standalone delta model from a converted checkpoint.

    `checkpoint` is None (auto: 'model_moe' if its weights are present in anysolv/models, else
    the included 'model_compact'), a checkpoint name, or a path to a converted .pt. Returns an
    AnySolvModel in eval mode on `device` with params cast to `dtype` (use torch.float64 for
    high-accuracy checks).

    `inference_settings` selects the inference path: a preset name -- 
    'default' (reference implementation), 
    'fast' (block-GEMM SO2 + tf32 + torch.compile, no MoE merging),
    'fast_gpu' (adds the Triton Wigner kernels; CUDA-only)
    or a custom InferenceSettings (whose `execution_mode` field picks the backend).

    'fast'/'umas_fast_pytorch' is composition-independent on the non-MoE 'model_compact'.
    On the MoE 'model_moe' it would need a MoE merge first, so it is downgraded to 'general' + tf32

    'fast_gpu'/'umas_fast_gpu' (requires CUDA, lmax==mmax==2, triton) auto-manages the merge by backbone:
    compact models' performance remains identical to fast;
    MoE checkpoints are MoE-merged so block-GEMM + Triton + torch.compile all apply, at the cost of
    locking to ONE composition/charge/spin/solvent
    """
    if checkpoint is None:
        checkpoint = default_checkpoint()

    settings = guess_inference_settings(inference_settings)

    path = _resolve(checkpoint)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    # Tag predates the AniSolv -> AnySolv rename; released checkpoints carry it.
    if ckpt.get("format") != "anisolv-ckpt-v1":
        raise ValueError(f"{path} is not an anisolv-ckpt-v1 checkpoint")

    cfg = dict(ckpt["backbone_config"])
    if cfg.get("dataset_list") is not None and cfg.get("dataset_mapping") is None:
        cfg["dataset_mapping"] = {name: name for name in cfg.pop("dataset_list")}
    cls_name = str(cfg.get("model", "")).rsplit(".", 1)[-1] or _DEFAULT_BACKBONE
    try:
        backbone_cls = _BACKBONES[cls_name]
    except KeyError:
        raise ValueError(
            f"{path}: unsupported backbone class {cls_name!r} (known: {sorted(_BACKBONES)})"
        ) from None

    if settings.execution_mode == "umas_fast_gpu":
        if "cuda" not in str(device):
            raise ValueError(
                f"execution_mode='umas_fast_gpu' requires a CUDA device (got device={device!r})."
            )
        want_merge = backbone_cls is eSCNMDMoeBackbone
        if settings.merge_mole != want_merge:
            settings = replace(settings, merge_mole=want_merge)

    if backbone_cls is eSCNMDMoeBackbone and not settings.merge_mole:
        if settings.execution_mode in ("umas_fast_pytorch", "umas_fast_gpu"):
            logging.warning(
                "%s on a MoE checkpoint (%s) needs a MoE merge (merge_mole=True / the 'fast_gpu' "
                "preset); falling back to the general backend (tf32 still applies).",
                settings.execution_mode, cls_name,
            )
            settings = replace(settings, execution_mode="general")
        if settings.compile:
            logging.warning(
                "torch.compile is not supported on an unmerged MoE checkpoint (%s): the MoE "
                "routing side-channel is not dynamo-safe across graph breaks. Disabling compile "
                "(tf32 still applies).", cls_name,
            )
            settings = replace(settings, compile=False)

    if settings.execution_mode not in _SUPPORTED_EXECUTION_MODES:
        raise NotImplementedError(
            f"execution_mode={settings.execution_mode!r} is not wired into the standalone loader "
            f"(supported: {sorted(_SUPPORTED_EXECUTION_MODES)})."
        )

    cfg.pop("model", None)  # class-path artifact, not a constructor kwarg
    cfg.update(_INFERENCE_OVERRIDES)
    cfg["execution_mode"] = settings.execution_mode  # ckpt backbone_config carries no such key

    backbone = backbone_cls(**cfg)
    head = MLP_EFS_Head(backbone)  # nulls backbone.energy_block/force_block internally
    model = AnySolvModel(backbone, head, ckpt["norm"], settings=settings)
    model.eps_transform = ckpt.get("eps_transform")
    if model.eps_transform is None:
        env = os.environ.get("ANYSOLV_SOLVENT_EPS_TRANSFORM")
        if env:
            import logging
            logging.warning("checkpoint %s lacks eps_transform; using env override %r",
                            path.name, env)
            model.eps_transform = env
        else:
            raise ValueError(
                f"checkpoint {path} carries no eps_transform tag. Re-tag it "
                f"(lr_augment/scripts/12_backfill_eps_tag.py for log-era files, or "
                f"reconvert via convert_checkpoint.py <in> <name> <log|born>), or set "
                f"ANYSOLV_SOLVENT_EPS_TRANSFORM explicitly.")
    if model.eps_transform not in ("log", "born"):
        raise ValueError(f"bad eps_transform {model.eps_transform!r} in {path}")

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    # Only non-persistent buffers (grid mats) may be "missing"; nothing should be unexpected.
    missing = [k for k in missing if "to_grid_mat" not in k and "from_grid_mat" not in k]
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch loading {path}:\n  missing={missing}\n  unexpected={unexpected}"
        )

    model.eval().to(device=device, dtype=dtype)
    return model
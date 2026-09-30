<h1 align="center">AnySolv</h1>
<p align="center"><em>Fast and Transferable MLIP Implicit Solvation</em></p>

> **Note:** AnySolv was formerly called **AniSolv** (package `anisolv`).

AnySolv predicts the single-point solvation energy: it produces a solvation energy $\Delta E$ and the associated force correction $\Delta F$, which can be added to any gas-phase potential (MLIP or DFT) to obtain solvated energies and forces. AnySolv is trained on molecular systems.

$$
\begin{aligned}
E_{\text{solvated}} &= E_{\text{gas}} + \Delta E_{\text{anysolv}} \\
F_{\text{solvated}} &= F_{\text{gas}} + \Delta F_{\text{anysolv}}
\end{aligned}
$$

where $\Delta E = E_{\text{solv}} - E_{\text{gas}}$. 

Key properties:
- **Can be paired with any base potential.** 
- **Lightweight at inference.** Models are optimized for inference speed.

### Model(s)

Two checkpoints are supported, in this order of preference: **`model_moe` > `model_compact`**. Both carry a per-system **solvent embedding** and an **output gate** (vacuum is exactly 0).
`load_model` / `predict_solvation_energy` auto-select the backbone from the checkpoint, so one can easily change models by specifying `checkpoint=`:

- **`model_moe`** (default when present) — a 64-expert MoE `eSCNMDMoeBackbone` (~291 M params), the full-accuracy model, trained against SMD reference solvation data. **Initialized from Meta's UMA `uma-s-1p2`**; the weights must be downloaded from Hugging Face (see below).
- **`model_compact`** — a dense GNN (~6.4 M params, ~25 MB) for fast / low-memory inference, with the same solvent conditioning and gate, trained against the same SMD reference data. Trained from scratch and already included in this repo.

Leaving `checkpoint` unset (`None`) auto-selects `model_moe` when its weights are present and otherwise falls back to the already included `model_compact`.

> **Deprecated:** all earlier checkpoints (the MoE `model1` and `model_smd`, and the compact `model1_compact` and `model_smd_compact`) are superseded by `model_moe` / `model_compact`, which outperform them across the board. They are unsupported; if you still have one, load it by explicit path (`checkpoint="/path/to/model.pt"`).

## Installation

```bash
git clone git@github.com:Ant-on-knee/anysolv.git
cd anysolv
pip install -e .              # editable install; or `pip install .`
```

You can also install straight from GitHub without cloning:

```bash
pip install "git+ssh://git@github.com/Ant-on-knee/anysolv.git"
```

Optional extras:

```bash
pip install -e ".[ase]"       # ASE - geometry I/O for the sample scripts (alias of [samples])
pip install -e ".[hub]"       # huggingface_hub - needed to download the weights (below)
```

The single_point sample runs without ASE; install the extra only if you want ASE's structures or `ase.Atoms` I/O. 
For a non-editable (PyPI) install, use `pip install "anysolv[ase]"`.

> **PyTorch note:** `pip` will pull a default `torch` build. For a specific CUDA/CPU build, install torch from [pytorch.org](https://pytorch.org/get-started/locally/) first, then install AnySolv.

## Download the model weights

The trained checkpoint **`model_moe.pt` (~1.1 GB) is not in this repository** - it is git-ignored and distributed separately on Hugging Face.

**1. Request access.** Go to **https://huggingface.co/antonknee/anysolv** and accept the FAIR Chemistry License. You must provide your full legal name, date of birth, and organization.

**2. Authenticate.**

```bash
pip install huggingface_hub
hf auth login                  # paste a token from https://huggingface.co/settings/tokens
```

**3. Download the checkpoint.**

- **If you cloned the repo and installed with `pip install -e .`**, drop it into the `models` directory:

  ```bash
  hf download antonknee/anysolv model_moe.pt --local-dir models
  ```

- **If you installed with a plain `pip install`** (from PyPI or `pip install git+…`), download it to any directory you control and pass its **absolute path** at call time. Until you do, the already included `model_compact` remains the default:

  ```bash
  hf download antonknee/anysolv model_moe.pt --local-dir /path/to/anysolv-weights
  ```

  ```python
  from anysolv import predict_solvation_energy
  predict_solvation_energy(..., checkpoint="/path/to/anysolv-weights/model_moe.pt")  # absolute path
  ```

> **License note:** these weights are a derivative of Meta's UMA (`uma-s-1p2`) and are governed by the **FAIR Chemistry License - not MIT**. The MIT license in this repo covers the *inference code only* and does not extend to the weights.

## Quickstart

```python
from anysolv import predict_solvation_energy

# Water geometry: atomic numbers Z and positions R (angstrom)
Z = [8, 1, 1]
R = [[0.0, 0.0,  0.119],
     [0.0, 0.763, -0.477],
     [0.0, -0.763, -0.477]]

# Solvent defaults to water; dE is in eV, dF in eV/angstrom (shape [n_atoms, 3]).
dE, dF = predict_solvation_energy((Z, R), charge=0, spin=1)
print(f"dE = {dE:.4f} eV")

# Vacuum baseline is exactly zero:
dE0, _ = predict_solvation_energy((Z, R), solvent=None)
assert dE0 == 0.0
```

## API

```python
predict_solvation_energy(
    atoms_or_arrays,           # ase.Atoms, or a (atomic_numbers, positions[angstrom]) tuple
    charge: int = 0,           # total charge
    spin: int = 1,             # spin multiplicity
    solvent="water",           # solvent name (str), or None for vacuum (-> exactly 0)
    checkpoint: str = None,    # auto: "model_moe" > "model_compact"; or a name / path to a .pt
    device: str = "cpu",       # "cpu", "cuda", or "mps"
    dtype=torch.float32,       # torch.float32 (default) or torch.float64
    inference_settings="default",  # "default" (reference), "fast", or "fast_gpu" (see below)
) -> tuple[float, np.ndarray]  # (dE in eV, dF in eV/angstrom with shape [n_atoms, 3])
```

To convert $\Delta E$ to kcal/mol, multiply by `23.060548`.

### Faster inference

`inference_settings` selects the compute path (also accepted by `load_model`):

- **`"default"`** — the pure-torch reference path. Bit-for-bit identical to earlier releases.
- **`"fast"`** — the block-diagonal SO2 GEMM backend plus TF32 matmuls and `torch.compile`.
  - Recommended for compact models (e.g. `model_compact`)
- **`"fast_gpu"`** — everything in `"fast"` **plus Triton Wigner kernels** (CUDA-only; requires `lmax==mmax==2`; install the optional `triton` via `pip install -e ".[gpu]"`, though it is already included in the CUDA `torch` wheels). The loader auto-manages the MoE merge by model:
  - Recommended for MoE models (e.g. `model_moe`)

```python
# compact, any molecule:
dE, dF = predict_solvation_energy((Z, R), checkpoint="model_compact",
                                  device="cuda", inference_settings="fast")
# MoE model_moe
dE, dF = predict_solvation_energy((Z, R), checkpoint="model_moe",
                                  device="cuda", inference_settings="fast_gpu")
```

For full control, pass an `InferenceSettings` instead of a preset name (e.g. merged block-GEMM
without Triton):

```python
from anysolv import InferenceSettings, predict_solvation_energy
settings = InferenceSettings(execution_mode="umas_fast_pytorch", tf32=True, compile=True,
                             merge_mole=True)  # MoE: merge -> block-GEMM + compile, single-composition
dE, dF = predict_solvation_energy((Z, R), checkpoint="model_moe", device="cuda",
                                  inference_settings=settings)
```

> **`torch.compile` caveat:** the first call is slow (graph capture) and a new molecule (element ratio) can trigger a recompile; if compilation fails the model falls back to eager automatically. TF32 and `torch.compile` mainly help on GPU. The Triton `umas_fast_gpu` backend is GPU-only and (on the MoE model) single-composition.

## Sample scripts

The sample scripts live in this repository (clone it to run them). From the repo root:

```bash
python anysolv/samples/H2O_single_point.py   # hydration dG for small molecules vs. experiment (ASE optional)
python anysolv/samples/H2O_dGsolv.py         # full thermodynamic cycle: geometry relax + vibrational dG (needs ASE + a loaded gas-phase MLIP)
```

Both auto-select the checkpoint (`model_moe` > `model_compact`); pass `--checkpoint` to pick one
explicitly (a name, or a path to a `.pt`) and `--device cpu|cuda|mps`:

```bash
python anysolv/samples/H2O_single_point.py --checkpoint model_compact
python anysolv/samples/H2O_dGsolv.py --checkpoint model_moe --device cuda
```

## Supported solvents

The model is trained/validated on 36 solvents:

water; acetone; acetonitrile; aniline; benzaldehyde; benzene; bromobenzene; carbon tetrachloride; dichloromethane; chloroform; chlorobenzene; carbon disulfide; cyclohexanone; 1,2-dichloroethane; diiodomethane; 1,4-dioxane; DMF; DMSO; ethanol; diethyl ether; ethyl acetate; n-hexadecane; n-hexane; iodobenzene; methanol; nitromethane; N-methylformamide; 1-octanol; o-dichlorobenzene; n-pentane; 1-pentanol; 1-propanol; 2,2,2-trifluoroethanol; THF; toluene; tributyl phosphate.

The model has been shown to extrapolate very well to untrained solvents, and all 179 solvents in the Minnesota Solvent Descriptor Database are supported.

## License

- **Inference code (this repository): MIT** - see [`LICENSE`](LICENSE). The included
  **`model_compact.pt`** weights are trained from scratch.
- **Full-accuracy weights (`model_moe.pt`, on Hugging Face): FAIR Chemistry License v1.** A derivative of Meta's UMA (`uma-s-1p2`); redistribution is permitted only under the same license. Use is subject to the FAIR Chemistry Acceptable Use Policy and applicable Trade Control Laws.

## Citation

If you use AnySolv, please cite both the UMA work it derives from and this repository (see also
[`CITATION.cff`](CITATION.cff)):

```bibtex
@article{wood2025uma,
  title   = {{UMA}: A Family of Universal Models for Atoms},
  author  = {Wood, Brandon M. and Dzamba, Misko and Fu, Xiang and Gao, Meng and Shuaibi, Muhammed and Barroso-Luque, Luis and Abdelmaqsoud, Kareem and Gharakhanyan, Vahe and Kitchin, John R. and Levine, Daniel S. and Michel, Kyle and Sriram, Anuroop and Cohen, Taco and Das, Abhishek and Rizvi, Ammar and Sahoo, Sushree Jagriti and Ulissi, Zachary W. and Zitnick, C. Lawrence},
  year    = {2025},
  journal = {arXiv preprint arXiv:2506.23971},
  doi     = {10.48550/arXiv.2506.23971},
  url     = {https://arxiv.org/abs/2506.23971}
}

@misc{anysolv2026,
  author       = {Ni, Anton Z.},
  title        = {{AnySolv: Fast and Transferable MLIP Implicit Solvation}},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Ant-on-knee/anysolv}},
  note         = {GitHub repository}
}
```

## Acknowledgements

`model_moe` is built by initializing weights from Meta FAIR Chemistry's UMA-S 1.2 (`uma-s-1p2`). 
UMA code is MIT-licensed ([facebookresearch/fairchem](https://github.com/facebookresearch/fairchem)); UMA weights are under the FAIR Chemistry License.

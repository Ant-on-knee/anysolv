"""Solvent conditioning vector for the solvent-embedding checkpoint variant.

get_solvent_vector(name) -> (1, 8) float32: seven vacuum-anchored normalized descriptors
(transform(raw)/scale, no mean subtraction, so gas phase = exactly 0 in every channel) + a
solvent-present mask. None / "vacuum" / "gas" -> the all-zero null vector.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import re
from pathlib import Path

import torch

_JSON_PATH = Path(__file__).resolve().parent / "_const" / "solvent_descriptors.json"

# Seven descriptors in the order SolventEmbedding expects (n25 excluded: ~collinear with n).
SOLVENT_DESCRIPTOR_ORDER = ["n", "alpha", "beta", "gamma", "epsilon", "aromaticity", "en-halogen"]
SOLVENT_DIM = len(SOLVENT_DESCRIPTOR_ORDER) + 1  # + solvent-present mask

# gas phase maps to 0 in every channel
# epsilon is scaled by log to mimic Born model (1-1/epsilon decays too quickly for large epsilon)
# refractive indices are shifted by 1
# must be consistent with fairchem-solvation fork
_SOLVENT_STATS = {
    "n": {"transform": "shift1", "scale": 0.068400},
    "alpha": {"transform": "linear", "scale": 0.181746},
    "beta": {"transform": "linear", "scale": 0.234892},
    "gamma": {"transform": "linear", "scale": 11.688257},
    "epsilon": {"transform": "born", "scale": 0.165059},
    "aromaticity": {"transform": "linear", "scale": 0.322355},
    "en-halogen": {"transform": "linear", "scale": 0.174984},
}

_EPS_STATS = {
    "log": {"transform": "log", "scale": 0.960804},
    "born": {"transform": "born", "scale": 0.165059},
}
_EPS_TRANSFORM_ENV = os.environ.get("ANYSOLV_SOLVENT_EPS_TRANSFORM", "log")
if _EPS_TRANSFORM_ENV not in _EPS_STATS:
    raise ValueError(f"ANYSOLV_SOLVENT_EPS_TRANSFORM must be one of "
                     f"{sorted(_EPS_STATS)}, got {_EPS_TRANSFORM_ENV!r}")
_SOLVENT_STATS["epsilon"] = dict(_EPS_STATS[_EPS_TRANSFORM_ENV])


def _transform(name: str, value: float) -> float:
    """Apply a descriptor's vacuum-anchoring transform (0 at the gas phase)."""
    transform = _SOLVENT_STATS[name]["transform"]
    if transform == "shift1":
        return float(value) - 1.0
    if transform == "log":
        return math.log(value)
    if transform == "born":
        return 1.0 - 1.0 / float(value)
    return float(value)

_VACUUM_NAMES = {"", "vacuum", "gas", "gas_phase", "gas-phase", "none"}

_ALIASES = {
    "carbontet": "carbon tetrachloride",
    "tbutylbenzene": "tert-butylbenzene",
    "phenylether": "diphenyl ether",
    "dibromoethane": "1,2-dibromoethane",
    "chlorohexane": "1-chlorohexane",
    "fluoroctane": "1-fluorooctane",
    "bromooctane": "1-bromooctane",
    "hexadecyliodide": "1-iodohexadecane",

    # defaults for ambiguous names 
    "trimethylbenzene": "1,2,4-trimethylbenzene", 
    "isopropyltoluene": "p-isopropyltoluene",
    "methylformamide": "N-methylformamide (E/Z mixture)",
    "tetrahydrothiophenedioxide": "tetrahydrothiophene-S,S-dioxide",
    "dimethylpyridine": "2,6-dimethylpyridine",

    # abbreviations
    "dmso": "dimethyl sulfoxide (DMSO)",
    "dimethylsulfoxide": "dimethyl sulfoxide (DMSO)",
    "dimethyl sulphoxide": "dimethyl sulfoxide (DMSO)",
    "dmf": "N,N-dimethylformamide",
    "dimethylformamide": "N,N-dimethylformamide",
    "dmac": "N,N-dimethylacetamide",
    "dimethylacetamide": "N,N-dimethylacetamide",
    "nmf": "N-methylformamide (E/Z mixture)",
    "thf": "tetrahydrofuran",
    "dcm": "dichloromethane",
    "methylene chloride": "dichloromethane",
    "dce": "1,2-dichloroethane",
    "dichloroethane": "1,2-dichloroethane",
    "ethylene dichloride": "1,2-dichloroethane",
    "ccl4": "carbon tetrachloride",
    "tetrachloromethane": "carbon tetrachloride",
    "cs2": "carbon disulfide",
    "meoh": "methanol",
    "etoh": "ethanol",
    "ipa": "2-propanol",
    "mecn": "acetonitrile",
    "acn": "acetonitrile",
    "h2o": "water",
    "tbp": "tributylphosphate",
    "tributyl phosphate": "tributylphosphate",
    "dipe": "diisopropyl ether",
    "mek": "butanone",
    "methyl ethyl ketone": "butanone",
    "2-butanone": "butanone",
    "mibk": "4-methyl-2-pentanone",
    "methyl isobutyl ketone": "4-methyl-2-pentanone",
    "etoac": "ethyl ethanoate",
    "sulfolane": "tetrahydrothiophene-S,S-dioxide",
    "tetramethylene sulfone": "tetrahydrothiophene-S,S-dioxide",
    "dioxane": "1,4-dioxane",
    "ether": "diethyl ether",
    "diethylether": "diethyl ether",
    "ethoxyethane": "diethyl ether",
    "di-n-butyl ether": "dibutyl ether",
    "isooctane": "2,2,4-trimethylpentane",

    # n-alkanes
    "pentane": "n-pentane",
    "hexane": "n-hexane",
    "heptane": "n-heptane",
    "octane": "n-octane",
    "nonane": "n-nonane",
    "decane": "n-decane",
    "undecane": "n-undecane",
    "dodecane": "n-dodecane",
    "pentadecane": "n-pentadecane",
    "hexadecane": "n-hexadecane",
    "butylbenzene": "n-butylbenzene",
    "cetane": "n-hexadecane",

    # alcohols: maps unspecified name to "n-" isomer, and the IUPAC "-an-N-ol" form
    "propanol": "1-propanol", "n-propanol": "1-propanol", "propan-1-ol": "1-propanol",
    "butanol": "1-butanol", "n-butanol": "1-butanol", "butan-1-ol": "1-butanol",
    "pentanol": "1-pentanol", "n-pentanol": "1-pentanol", "pentan-1-ol": "1-pentanol",
    "amyl alcohol": "1-pentanol",
    "hexanol": "1-hexanol", "hexan-1-ol": "1-hexanol",
    "heptanol": "1-heptanol", "heptan-1-ol": "1-heptanol",
    "octanol": "1-octanol", "octan-1-ol": "1-octanol",
    "nonanol": "1-nonanol", "nonan-1-ol": "1-nonanol",
    "decanol": "1-decanol", "decan-1-ol": "1-decanol",
    "isopropanol": "2-propanol", "isopropyl alcohol": "2-propanol",
    "propan-2-ol": "2-propanol",
    "sec-butanol": "2-butanol", "s-butanol": "2-butanol", "butan-2-ol": "2-butanol",
    "tert-butanol": "2-methyl-2-propanol", "t-butanol": "2-methyl-2-propanol",
    "tert-butyl alcohol": "2-methyl-2-propanol",
    "2-methylpropan-2-ol": "2-methyl-2-propanol",
    "isobutanol": "2-methyl-1-propanol", "isobutyl alcohol": "2-methyl-1-propanol",
    "2-methylpropan-1-ol": "2-methyl-1-propanol",
    "ethylene glycol": "1,2-ethanediol", "ethane-1,2-diol": "1,2-ethanediol",
    "monoethylene glycol": "1,2-ethanediol", "glycol": "1,2-ethanediol",
    "allyl alcohol": "2-propen-1-ol", "prop-2-en-1-ol": "2-propen-1-ol",
    "trifluoroethanol": "2,2,2-trifluoroethanol", "tfe": "2,2,2-trifluoroethanol",
    "methoxyethanol": "2-methoxyethanol", "methyl cellosolve": "2-methoxyethanol",

    # esters
    "methyl acetate": "methyl ethanoate",
    "ethyl acetate": "ethyl ethanoate",
    "propyl acetate": "propyl ethanoate", "n-propyl acetate": "propyl ethanoate",
    "butyl acetate": "butyl ethanoate", "n-butyl acetate": "butyl ethanoate",
    "pentyl acetate": "pentyl ethanoate", "amyl acetate": "pentyl ethanoate",
    "methyl formate": "methyl methanoate",
    "ethyl formate": "ethyl methanoate",
    "methyl propionate": "methyl propanoate",
    "methyl butyrate": "methyl butanoate",

    # ketones
    "pentan-2-one": "2-pentanone",
    "pentan-3-one": "3-pentanone", "diethyl ketone": "3-pentanone",
    "hexan-2-one": "2-hexanone",
    "heptan-2-one": "2-heptanone",
    "octan-2-one": "2-octanone",
    "heptan-4-one": "4-heptanone", "dipropyl ketone": "4-heptanone",
    "nonan-5-one": "5-nonanone", "dibutyl ketone": "5-nonanone",
    "4-methylpentan-2-one": "4-methyl-2-pentanone",

    # nitriles
    "propionitrile": "propanonitrile", "propiononitrile": "propanonitrile",
    "butyronitrile": "butanonitrile",

    # acids and aldehydes
    "ethanoic acid": "acetic acid",
    "methanoic acid": "formic acid",
    "propionic acid": "propanoic acid",
    "butyric acid": "butanoic acid",
    "valeric acid": "pentanoic acid",
    "caproic acid": "hexanoic acid",
    "propionaldehyde": "propanal",
    "butyraldehyde": "butanal",
    "valeraldehyde": "pentanal",

    # amines and aromatics
    "phenylamine": "aniline", "aminobenzene": "aniline",
    "n-butylamine": "butylamine",
    "n-propylamine": "propylamine",
    "n-pentylamine": "pentylamine",
    "2-picoline": "2-methylpyridine", "alpha-picoline": "2-methylpyridine",
    "3-picoline": "3-methylpyridine", "beta-picoline": "3-methylpyridine",
    "4-picoline": "4-methylpyridine", "gamma-picoline": "4-methylpyridine",
    "2,4-lutidine": "2,4-dimethylpyridine",
    "2,6-lutidine": "2,6-dimethylpyridine",
    "cumene": "isopropylbenzene",
    "p-cymene": "p-isopropyltoluene",
    "1,3,5-trimethylbenzene": "mesitylene",
    "pseudocumene": "1,2,4-trimethylbenzene",
    "methoxybenzene": "anisole",
    "phenetole": "ethyl phenyl ether", "ethoxybenzene": "ethyl phenyl ether",
    "benzyl chloride": "alpha-chlorotoluene",
    "hexafluorobenzene": "perfluorobenzene",
    "1,2,3,4-tetrahydronaphthalene": "tetralin",
    "decahydronaphthalene": "decalin (cis/trans mixture)",
    "2-methylphenol": "o-cresol",
    "3-methylphenol": "m-cresol",
    "2-nitrotoluene": "o-nitrotoluene",
    "2-chlorotoluene": "o-chlorotoluene",
    "1,2-dichlorobenzene": "o-dichlorobenzene",

    # halogenated
    "trichloroethylene": "trichloroethene",
    "tetrachloroethylene": "tetrachloroethene",
    "perchloroethylene": "tetrachloroethene",
    "trichloromethane": "chloroform",
    "tribromomethane": "bromoform",
    "methyl iodide": "iodomethane",
    "ethyl iodide": "iodoethane",
    "methylene iodide": "diiodomethane",
    "methylene bromide": "dibromomethane",
    "ethyl bromide": "bromoethane",
    "methyl chloroform": "1,1,1-trichloroethane",
    "ethylene dibromide": "1,2-dibromoethane",
    "isobutyl bromide": "1-bromo-2-methylpropane",
    "isopropyl bromide": "2-bromopropane",
    "n-propyl bromide": "1-bromopropane",
    "trans-1,2-dichloroethene": "E-1,2-dichloroethene",
    "cis-1,2-dichloroethene": "Z-1,2-dichloroethene",

    # sulfur
    "ethyl mercaptan": "ethanethiol",

    # alkenes
    "trans-2-pentene": "E-2-pentene",
}

_RACEMIC_PREFIXES = ("(+/-)-", "(+-)-", "(±)-", "(rs)-", "rac-", "dl-", "d,l-", "(\\xb1)-", "(\\xb1)")


@functools.lru_cache(maxsize=1)
def _load_raw() -> dict:
    with _JSON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _norm(name) -> str:
    """Fold a user-supplied solvent name to its lookup form.

    Case, surrounding/repeated whitespace and a racemic prefix carry no chemical meaning,
    so none of them should decide whether a lookup succeeds.
    """
    s = " ".join(str(name).strip().lower().split())
    for pre in _RACEMIC_PREFIXES:
        if s.startswith(pre):
            return s[len(pre):].lstrip("- ")
    return s


@functools.lru_cache(maxsize=1)
def _solvents_ci() -> dict:
    """Normalized index of the solvent table (some keys carry uppercase, e.g.
    'dimethyl sulfoxide (DMSO)', 'N,N-dimethylformamide', 'E-1,2-dichloroethene')."""
    return {_norm(k): v for k, v in _load_raw()["solvents"].items()}


def _squash(s: str) -> str:
    """Lowercase with every separator removed. Not a normaliser -- see _squash_ci."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


@functools.lru_cache(maxsize=1)
def _canon_ci() -> dict:
    """Normalized name -> the table's own spelling of it."""
    return {_norm(k): k for k in _load_raw()["solvents"]}


@functools.lru_cache(maxsize=1)
def _alias_ci() -> dict:
    """Normalized alias -> canonical table key, dropping any alias that would shadow a
    real solvent name (a table name must always win) or point at a missing key."""
    table = _load_raw()["solvents"]
    real = {_norm(k) for k in table}
    out = {}
    for alias, target in _ALIASES.items():
        a = _norm(alias)
        if a in real or target not in table:
            continue
        out[a] = target
    for key in table:
        stem = _norm(re.sub(r"\s*\([^)]*\)\s*$", "", key))
        if stem and stem not in real and stem not in out:
            out[stem] = key
    return out


@functools.lru_cache(maxsize=1)
def _squash_ci() -> dict:
    """Separator-free name -> canonical key, for source tables that omit punctuation.
    """
    table = _load_raw()["solvents"]
    hits: dict = {}
    for k in table:
        hits.setdefault(_squash(k), set()).add(k)
        stem = re.sub(r"\s*\([^)]*\)\s*$", "", k)
        hits.setdefault(_squash(stem), set()).add(k)
    for a, target in _alias_ci().items():
        hits.setdefault(_squash(a), set()).add(target)
    return {s: next(iter(v)) for s, v in hits.items() if s and len(v) == 1}


def resolve_solvent_name(name):
    """Canonical table key for a solvent name or accepted alias; None if unrecognized.

    Case-insensitive. Returns "" for vacuum/gas-phase and None for unknown names, so
    callers can tell "explicitly no solvent" from "I don't know this one".
    """
    if name is None:
        return ""
    key = _norm(name)
    if key in _VACUUM_NAMES:
        return ""
    canon = _canon_ci().get(key)
    if canon is not None:
        return canon
    alias = _alias_ci().get(key)
    if alias is not None:
        return alias
    return _squash_ci().get(_squash(key))


def list_solvents() -> list:
    return sorted(_load_raw()["solvents"].keys())


def list_aliases() -> dict:
    """Accepted alias -> canonical solvent name."""
    return dict(sorted(_alias_ci().items()))


def normalize(raw_vec, eps_transform: str | None = None) -> list:
    """Vacuum-anchored normalization; eps_transform picks the epsilon channel's
    stats ('log'/'born'; None -> the env/module default)."""
    if len(raw_vec) != len(SOLVENT_DESCRIPTOR_ORDER):
        raise ValueError(f"raw_vec must have {len(SOLVENT_DESCRIPTOR_ORDER)} values, got {len(raw_vec)}")
    stats = dict(_SOLVENT_STATS)
    if eps_transform is not None:
        if eps_transform not in _EPS_STATS:
            raise ValueError(f"eps_transform must be one of {sorted(_EPS_STATS)}, "
                             f"got {eps_transform!r}")
        stats["epsilon"] = _EPS_STATS[eps_transform]
    out = []
    for name, value in zip(SOLVENT_DESCRIPTOR_ORDER, raw_vec):
        tf = stats[name]["transform"]
        if tf == "shift1":
            v = float(value) - 1.0
        elif tf == "log":
            v = math.log(value)
        elif tf == "born":
            v = 1.0 - 1.0 / float(value)
        else:
            v = float(value)
        out.append(v / stats[name]["scale"])
    return out


def get_solvent_vector(solvent_name, strict: bool = True,
                       eps_transform: str | None = None) -> torch.Tensor:
    """Build the (1, SOLVENT_DIM) conditioning vector. None/vacuum -> null vector.

    eps_transform: 'log' (r4 and earlier) / 'born' (r5+). Pass the CHECKPOINT'S
    value (convert_checkpoint.py bakes it; load_model exposes model.eps_transform);
    None falls back to ANYSOLV_SOLVENT_EPS_TRANSFORM, default 'log'.
    """
    vec = torch.zeros(1, SOLVENT_DIM, dtype=torch.float32)
    if solvent_name is None:
        return vec
    key = _norm(solvent_name)
    if key in _VACUUM_NAMES:
        return vec
    solvents = _solvents_ci()
    if key not in solvents:
        alias = _alias_ci().get(key)
        if alias is not None:
            key = _norm(alias)
    if key not in solvents:
        if strict:
            raise KeyError(
                f"Unknown solvent '{solvent_name}'. Use strict=False for the vacuum "
                f"vector, or see list_solvents() / list_aliases()."
            )
        logging.warning("Unknown solvent '%s'; using the vacuum vector.", solvent_name)
        return vec
    raw = [solvents[key][name] for name in SOLVENT_DESCRIPTOR_ORDER]
    vec[0, : len(SOLVENT_DESCRIPTOR_ORDER)] = torch.tensor(
        normalize(raw, eps_transform=eps_transform), dtype=torch.float32)
    vec[0, len(SOLVENT_DESCRIPTOR_ORDER)] = 1.0
    return vec

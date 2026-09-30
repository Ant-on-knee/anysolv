"""Guards on the solvent-name alias layer (no checkpoint or GPU required).

    python -m pytest anysolv/tests/test_solvent_aliases.py -q

The alias table is hand-maintained; this unittest tries to mitigate typographical mistakes.
"""

import torch

from anysolv.solvent import (SOLVENT_DIM, _ALIASES, _load_raw, _norm, get_solvent_vector,
                             list_aliases, list_solvents, resolve_solvent_name)


def _table():
    return _load_raw()["solvents"]


def test_every_alias_target_exists():
    missing = {a: t for a, t in _ALIASES.items() if t not in _table()}
    assert not missing, f"alias targets absent from the descriptor table: {missing}"


def test_no_alias_shadows_a_real_solvent():
    real = {_norm(k) for k in _table()}
    assert not [a for a in _ALIASES if _norm(a) in real]


def test_no_conflicting_duplicate_aliases():
    seen, conflicts = {}, []
    for alias, target in _ALIASES.items():
        n = _norm(alias)
        if seen.get(n, target) != target:
            conflicts.append((alias, seen[n], target))
        seen[n] = target
    assert not conflicts


def test_lookup_is_case_and_whitespace_insensitive():
    for name in ("water", "DMSO", "hexane", "ethyl acetate", "sulfolane", "THF"):
        want = resolve_solvent_name(name)
        assert want is not None, name
        for variant in (name.upper(), name.lower(), name.title(),
                        f"  {name}  ", name.replace(" ", "  ")):
            assert resolve_solvent_name(variant) == want, variant


def test_canonical_names_all_resolve_to_themselves():
    for key in list_solvents():
        assert resolve_solvent_name(key) == key


def test_alias_and_canonical_give_the_identical_vector():
    for alias, target in list_aliases().items():
        assert torch.equal(get_solvent_vector(alias), get_solvent_vector(target)), alias


def test_racemic_prefix_is_stripped_but_enantiomers_are_not():
    assert resolve_solvent_name("(+/-)-2-butanol") == "2-butanol"
    assert resolve_solvent_name("(±)-2-butanol") == "2-butanol"
    # A single enantiomer must NOT silently inherit racemate descriptors.
    assert resolve_solvent_name("(R)-2-butanol") is None


def test_vacuum_and_unknown_behaviour_is_unchanged():
    assert resolve_solvent_name(None) == ""
    for name in ("", "vacuum", "gas", "gas-phase", "none"):
        assert resolve_solvent_name(name) == ""
        assert float(get_solvent_vector(name).abs().sum()) == 0.0
    assert resolve_solvent_name("nonesuch") is None
    try:
        get_solvent_vector("nonesuch")
    except KeyError:
        pass
    else:
        raise AssertionError("strict lookup must raise on an unknown solvent")
    vec = get_solvent_vector("nonesuch", strict=False)
    assert vec.shape == (1, SOLVENT_DIM)
    assert float(vec.abs().sum()) == 0.0

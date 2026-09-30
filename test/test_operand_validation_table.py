"""W03 operand-validation table (2026-09-24): every pointer operand of every
algorithm in the ABI table belongs to one documented operand class, and the
concepts page lists every class. A new algorithm with an unlisted operand fails
here on the CPU lane, before it can reach a kernel with an unchecked shape."""
from __future__ import annotations

from pathlib import Path

from grid_codegen.abi_specs import ABI_SPECS

_REPO = Path(__file__).resolve().parents[1]
_PAGE = _REPO / "docs/source/user_guide/concepts/operand_validation.rst"

# operand name -> class (mirrors the table on the concepts page)
OPERAND_CLASSES = {
    "q": "configuration", "qd": "configuration", "qdd": "configuration",
    "qdd_opt": "configuration", "u": "configuration", "var": "configuration",
    "lower": "configuration", "upper": "configuration", "u_des": "configuration",
    "f_ext": "force",
    "x": "state", "x_des": "state",
    "p_des": "plant operand", "W": "plant operand", "Q": "plant operand",
    "R": "plant operand", "h_des": "plant operand",
    "wrench": "tool operand", "rc": "tool operand",
    "offset": "runtime offset",
}
CLASSES = ("configuration", "force", "state", "plant operand", "tool operand",
           "runtime offset", "scalar")


def _pointer_operands():
    names = set()
    for spec in ABI_SPECS.values():
        for n, ty in spec.inputs:
            if ty == "const T*":
                names.add(n)
    return names


def test_every_pointer_operand_has_a_class():
    unlisted = _pointer_operands() - set(OPERAND_CLASSES)
    assert not unlisted, f"operands with no validation class: {sorted(unlisted)}"


def test_every_class_member_exists_in_the_abi_table():
    stale = set(OPERAND_CLASSES) - _pointer_operands()
    assert not stale, f"table lists operands the ABI no longer has: {sorted(stale)}"


def test_target_jid_is_a_trailing_runtime_arg_where_offsets_are():
    for key, spec in ABI_SPECS.items():
        names = {n for n, _ in spec.inputs}
        if "offset" in names:
            assert "target_jid" in spec.trailing_runtime_args, key


def test_concepts_page_lists_every_class_and_member():
    text = _PAGE.read_text()
    for c in CLASSES:
        assert f"``{c}``" in text, f"class {c!r} missing from {_PAGE.name}"
    for n in OPERAND_CLASSES:
        assert f"``{n}``" in text, f"operand {n!r} missing from {_PAGE.name}"
    assert "``target_jid``" in text

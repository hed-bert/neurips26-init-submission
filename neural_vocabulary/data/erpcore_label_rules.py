"""Per-paradigm label rules for ERP-CORE classical-ERP probes.

Single source of truth for mapping ERP-CORE BIDS event metadata
(``event_value`` + ``event_type`` from the preprocessed h5 attrs)
to per-paradigm probe labels. Centralized here so the probe scripts,
dataset code, and tests never drift on the value-code semantics
(verified against the 6 NEMAR ``task-*_events.json`` sidecars).

A LabelRule maps one (event_value, event_type) row to either an
integer class label (for that probe) or None (skip; this trial is
not part of the probe's labeled set).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class LabelRule:
    """A single per-paradigm classification probe.

    Attributes:
        paradigm: ERP-CORE task name (e.g. "N170", "P3").
        probe: short probe name (e.g. "face_vs_car", "target_vs_nontarget").
        n_classes: number of label classes (binary -> 2, etc.).
        class_names: human-readable names indexed by label int.
        label_fn: ``(event_value: str, event_type: str) -> int | None``;
            None means skip the trial (not part of this probe's set).
    """

    paradigm: str
    probe: str
    n_classes: int
    class_names: tuple[str, ...]
    label_fn: Callable[[str, str], int | None]

    def __post_init__(self) -> None:
        if not self.paradigm:
            raise ValueError("paradigm must be non-empty")
        if not self.probe:
            raise ValueError("probe must be non-empty")
        if self.n_classes < 2:
            raise ValueError(f"n_classes must be ≥ 2, got {self.n_classes}")
        if len(self.class_names) != self.n_classes:
            raise ValueError(
                f"{self.paradigm}/{self.probe}: len(class_names)="
                f"{len(self.class_names)} != n_classes={self.n_classes}"
            )

    def label(self, event_value: str, event_type: str) -> int | None:
        """Apply ``label_fn`` and validate the output range.

        Catches the two highest-impact bugs a future contributor can
        introduce: an off-by-one return value, and a positional swap of
        the two string args (``label_fn(et, v)`` instead of ``(v, et)``)
        that silently produces wrong labels rather than raising.
        """
        out = self.label_fn(event_value, event_type)
        if out is None:
            return None
        if not (0 <= out < self.n_classes):
            raise RuntimeError(
                f"{self.paradigm}/{self.probe}: label_fn returned {out}, "
                f"outside [0, {self.n_classes}) for "
                f"(event_value={event_value!r}, event_type={event_type!r})"
            )
        return out


# -----------------------------------------------------------------------------
# N170 (face perception) — labels keyed by event_type column.
#   "face" / "car" / "scrambled_face" / "scrambled_car" -- 80 trials each.
# -----------------------------------------------------------------------------


def _n170_face_vs_car(event_value: str, event_type: str) -> int | None:
    if event_type == "face":
        return 0
    if event_type == "car":
        return 1
    return None


def _n170_face_vs_scrambled_face(event_value: str, event_type: str) -> int | None:
    if event_type == "face":
        return 0
    if event_type == "scrambled_face":
        return 1
    return None


def _n170_object_3way(event_value: str, event_type: str) -> int | None:
    """Object-vs-texture 3-way: face=0, car=1, scrambled (either)=2."""
    if event_type == "face":
        return 0
    if event_type == "car":
        return 1
    if event_type in ("scrambled_face", "scrambled_car"):
        return 2
    return None


# -----------------------------------------------------------------------------
# MMN (passive auditory oddball) — labels keyed by event_value.
#   70 -> deviant, 80 -> standard, 180 -> initial-stream standard (skip:
#   context-establishing burst, not part of the oddball contrast).
# -----------------------------------------------------------------------------


def _mmn_standard_vs_deviant(event_value: str, event_type: str) -> int | None:
    if event_value == "80":
        return 0  # standard
    if event_value == "70":
        return 1  # deviant
    # 180 (initial-stream standards) and STATUS markers are skipped.
    return None


# -----------------------------------------------------------------------------
# P3 (active visual oddball) — labels keyed by event_value.
#   value = (letter_idx * 10) + block_target_idx, both in 1..5. Target trial
#   when letter_idx == block_target_idx, e.g. 11 (A in block A), 22, 33, 44, 55.
# -----------------------------------------------------------------------------


def _p3_target_vs_nontarget(event_value: str, event_type: str) -> int | None:
    if not event_value.isdigit():
        return None
    n = int(event_value)
    if not (10 <= n <= 99):  # stim codes only; 201/202 are responses
        return None
    letter_idx, block_target_idx = divmod(n, 10)
    if letter_idx < 1 or letter_idx > 5 or block_target_idx < 1 or block_target_idx > 5:
        return None
    return 0 if letter_idx != block_target_idx else 1


# -----------------------------------------------------------------------------
# N2pc (visual search) — labels keyed by event_value.
#   value digit pattern: hundreds=target_color (1=blue, 2=pink),
#                        tens=target_hemifield (1=left, 2=right),
#                        ones=gap_position (1=top, 2=bottom).
#   Stim codes: {111,112,121,122,211,212,221,222}. Responses: 201/202.
# -----------------------------------------------------------------------------


def _n2pc_target_hemifield(event_value: str, event_type: str) -> int | None:
    if not event_value.isdigit():
        return None
    n = int(event_value)
    if n not in {111, 112, 121, 122, 211, 212, 221, 222}:
        return None
    hemifield = (n // 10) % 10  # 1=left, 2=right
    if hemifield == 1:
        return 0
    if hemifield == 2:
        return 1
    return None


def _n2pc_contra_ipsi(event_value: str, event_type: str) -> int | None:
    """N2pc contra-ipsi hemifield direction label (Option B).

    Label encodes which hemifield contains the target:
        0 = left target  (contralateral electrode is right-hemisphere PO8)
        1 = right target (contralateral electrode is left-hemisphere PO7)

    This is the same hemifield logic as ``_n2pc_target_hemifield`` but with
    different semantic intent: downstream evaluation MUST compute the
    contra-minus-ipsi PO7/PO8 difference conditioned on this label rather
    than using absolute per-channel power.  Specifically:
      - label 0 (left target): contra-ipsi = PO8_power - PO7_power
      - label 1 (right target): contra-ipsi = PO7_power - PO8_power

    The N2pc effect is positive in the contralateral hemisphere (Luck & Hillyard
    1994), so a model that learns the correct TF-grounded representation should
    show higher power at PO8 for left targets and higher at PO7 for right
    targets in the N2pc time-frequency window (~180-280 ms, alpha/beta band).

    Implementation note (Option B rationale): ``LabelRule.label_fn`` receives
    only (event_value, event_type) -- no TF data.  Computing the actual signed
    PO7-PO8 difference requires raw trial data and therefore cannot live inside
    label_fn.  Option B keeps this rule strictly label-engineering (same input
    API, correct semantic labeling) and delegates TF feature extraction to the
    evaluation script that consumes the rule, where channel indices are known.
    """
    if not event_value.isdigit():
        return None
    n = int(event_value)
    if n not in {111, 112, 121, 122, 211, 212, 221, 222}:
        return None
    hemifield = (n // 10) % 10  # 1=left, 2=right
    if hemifield == 1:
        return 0  # left target: contra = PO8 (right hemisphere)
    if hemifield == 2:
        return 1  # right target: contra = PO7 (left hemisphere)
    return None


# -----------------------------------------------------------------------------
# N400 (semantic priming) — labels keyed by event_value.
#   1xx = prime word (red), 2xx = target word (green; the N400 elicitor).
#   For N400 we probe related/unrelated on TARGETS only.
#   211 = related list 1 target, 212 = related list 2,
#   221 = unrelated list 1 target, 222 = unrelated list 2.
# -----------------------------------------------------------------------------


def _n400_related_vs_unrelated_target(event_value: str, event_type: str) -> int | None:
    if event_value not in {"211", "212", "221", "222"}:
        return None
    return 0 if event_value in {"211", "212"} else 1


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------


LABEL_RULES: dict[str, list[LabelRule]] = {
    "N170": [
        LabelRule("N170", "face_vs_car", 2, ("face", "car"), _n170_face_vs_car),
        LabelRule(
            "N170",
            "face_vs_scrambled_face",
            2,
            ("face", "scrambled_face"),
            _n170_face_vs_scrambled_face,
        ),
        LabelRule(
            "N170",
            "object_3way",
            3,
            ("face", "car", "scrambled"),
            _n170_object_3way,
        ),
    ],
    "MMN": [
        LabelRule(
            "MMN",
            "standard_vs_deviant",
            2,
            ("standard", "deviant"),
            _mmn_standard_vs_deviant,
        ),
    ],
    "P3": [
        LabelRule(
            "P3",
            "target_vs_nontarget",
            2,
            ("non_target", "target"),
            _p3_target_vs_nontarget,
        ),
    ],
    "N2pc": [
        LabelRule(
            "N2pc",
            "target_hemifield",
            2,
            ("left", "right"),
            _n2pc_target_hemifield,
        ),
        LabelRule(
            "N2pc",
            "contra_ipsi",
            2,
            ("left_target_contra_PO8", "right_target_contra_PO7"),
            _n2pc_contra_ipsi,
        ),
    ],
    "N400": [
        LabelRule(
            "N400",
            "related_vs_unrelated",
            2,
            ("related", "unrelated"),
            _n400_related_vs_unrelated_target,
        ),
    ],
}


def get_rules_for_task(task: str) -> list[LabelRule]:
    """Return all label rules for an ERP-CORE task; empty list if unknown."""
    return LABEL_RULES.get(task, [])


def get_rule(paradigm: str, probe: str) -> LabelRule:
    """Return a single rule by (paradigm, probe) name; raises if not found."""
    for r in LABEL_RULES.get(paradigm, []):
        if r.probe == probe:
            return r
    raise KeyError(f"No LabelRule for paradigm={paradigm!r} probe={probe!r}")

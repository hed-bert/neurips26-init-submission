"""ERP-CORE event classifier.

Maps an ERP-CORE BIDS events.tsv row to one of the V9 Tier 1 epoch
windows (`stim` / `response`) consumed by `epoch_data` from
`scripts/preprocess_hbn.py`. The BIDS `trial_type` column is the
canonical source — it is unambiguous across the 6 ERP-CORE tasks
(flankers, MMN, N170, N2pc, N400, P3) where `value` codes overlap
across paradigms.
"""

from __future__ import annotations

_STIM_TRIAL_TYPES: frozenset[str] = frozenset({"stimulus"})
_RESPONSE_TRIAL_TYPES: frozenset[str] = frozenset({"response"})
# STATUS markers (e.g. MMN start markers, value=1 or 4) carry no
# trial-level meaning; n/a and empty strings are BIDS missing-data
# markers. Everything else falls through to a return-None skip.
_SKIP_TRIAL_TYPES: frozenset[str] = frozenset({"STATUS", "n/a", "nan", ""})


def classify_erpcore_event(trial_type: str | None) -> str | None:
    """Classify an ERP-CORE event into a V9 Tier 1 epoch type.

    Args:
        trial_type: value of the BIDS `trial_type` column for this event.

    Returns:
        ``"stim"`` for stimulus presentations,
        ``"response"`` for button presses,
        ``None`` for STATUS markers, missing values, or anything that
        cannot be classified (caller should skip the row).
    """
    if trial_type is None:
        return None
    tt = str(trial_type).strip()
    if tt in _SKIP_TRIAL_TYPES:
        return None
    if tt in _STIM_TRIAL_TYPES:
        return "stim"
    if tt in _RESPONSE_TRIAL_TYPES:
        return "response"
    return None

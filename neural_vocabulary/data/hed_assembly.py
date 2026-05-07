"""Public HED assembly utilities.

Shared module so cross-dataset HED pipelines (HBN, ERP-CORE, and any
future union corpora) import a single source of truth instead of
reaching into a script's private API.

The function is intentionally lightweight (no ``HEDVectorizer`` instance
needed) so it can be passed across process boundaries by
``ProcessPoolExecutor`` workers.
"""

from __future__ import annotations

import numpy as np


def vectorize_hed_string(
    hed_string: str,
    tag_to_idx: dict[str, int],
    vocab_size: int,
    schema_version: str = "8.3.0",
) -> np.ndarray:
    """Convert a HED string to a multi-hot ancestor-inclusive vector.

    Expands the string to long form via the HED schema, then sets bits
    for every long-form path *and all ancestors* present in
    ``tag_to_idx``. Callers must pre-expand ``Def/`` references before
    passing the string here; this function uses an empty
    ``DefinitionDict`` so ``expand_defs()`` is a defensive no-op.

    Args:
        hed_string: A HED annotation string. Caller is responsible for
            pre-expanding ``Def/`` references (see ``preprocess_hbn``
            for the BIDS sidecar substitution path).
        tag_to_idx: Vocabulary mapping (long-form path -> index).
        vocab_size: Dimensionality of the output vector.
        schema_version: HED schema version to load. Defaults to
            ``"8.3.0"`` for backwards-compat with HBN preprocessing.
            Callers from datasets that declare a different HED version
            (e.g. ERP-CORE's ``"8.4.0"``) should pass that version
            explicitly so the on-disk provenance attribute matches the
            schema actually used.

    Returns:
        ``np.ndarray`` of shape ``(vocab_size,)`` and dtype ``float32``,
        with 1.0 at every position whose ancestor chain is reachable
        from the expanded HED string.
    """
    from hed import HedString, load_schema_version
    from hed.models import DefinitionDict

    schema = load_schema_version(schema_version)
    # Def/ references should already be expanded by the caller.
    # The empty DefinitionDict makes expand_defs() a no-op here.
    hs = HedString(hed_string, schema, DefinitionDict())
    hs.expand_defs()
    all_tags = hs.get_all_tags()

    vec = np.zeros(vocab_size, dtype=np.float32)
    for tag in all_tags:
        long = tag.long_tag
        if "Def-expand" in long or "Definition" in long:
            continue
        if long.startswith("Property/Organizational-property/Def/"):
            continue
        # Set bits for this tag and all ancestors.
        parts = long.split("/")
        for depth in range(1, len(parts) + 1):
            ancestor = "/".join(parts[:depth])
            idx = tag_to_idx.get(ancestor)
            if idx is not None:
                vec[idx] = 1.0

    return vec

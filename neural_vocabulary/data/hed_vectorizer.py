"""HED tag vectorization for hierarchical event prediction.

Converts HED annotation strings into multi-hot binary vectors over a
shared tag vocabulary, with ancestor inclusion and depth weights for the
depth-weighted BCE loss. Uses hedtools for definition resolution and
long-form expansion.

Usage:
    vectorizer = HEDVectorizer(schema_version="8.3.0")
    vectorizer.build_vocabulary(all_hed_strings)
    vector = vectorizer.vectorize("(Agent-action, Rest)")
    weights = vectorizer.get_depth_weights(alpha=0.7)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch
from hed import HedString, load_schema_version
from hed.models import DefinitionDict

logger = logging.getLogger(__name__)

# Process-level HED branch prefixes for Gate 1b (process-HED).
# Includes cognitive-process branches shared across paradigms.
# Excludes: Item/* (content), Property/Sensory-property/Sensory-attribute/* (perceptual content).
PROCESS_BRANCHES: list[str] = [
    "Event/Sensory-event",
    "Event/Agent-action",
    "Event/Data-feature",
    "Property/Agent-property/Agent-state",
    "Event/Measurement-event",
    "Event/Experiment-structure",
    "Property/Task-property",
]


class HEDVectorizer:
    """Convert HED annotation strings to multi-hot vectors with hierarchy.

    The vectorizer:
    1. Resolves Def/ references against provided definitions
    2. Expands tags to long form (full hierarchy paths)
    3. Includes all ancestor tags (predicting a leaf implies predicting root)
    4. Computes per-tag depth weights for the depth-weighted BCE loss
    """

    def __init__(self, schema_version: str = "8.3.0", max_tag_depth: int = 0) -> None:
        self._schema = load_schema_version(schema_version)
        self._tag_to_idx: dict[str, int] = {}
        self._idx_to_tag: dict[int, str] = {}
        self._tag_depths: dict[str, int] = {}
        self._tag_doc_freq: dict[str, int] = {}
        self._n_docs: int = 0
        self._max_tag_depth: int = max_tag_depth
        self._branch_filter: list[str] = []
        self._def_dict: DefinitionDict = DefinitionDict()

    @property
    def vocab_size(self) -> int:
        """Number of unique tags in the vocabulary."""
        return len(self._tag_to_idx)

    @property
    def tag_to_idx(self) -> dict[str, int]:
        """Mapping from tag string to vocabulary index."""
        return dict(self._tag_to_idx)

    @property
    def idx_to_tag(self) -> dict[int, str]:
        """Mapping from vocabulary index to tag string."""
        return dict(self._idx_to_tag)

    @property
    def tag_depths(self) -> dict[str, int]:
        """Mapping from tag string to its depth in the HED hierarchy."""
        return dict(self._tag_depths)

    def load_definitions_from_sidecar(self, sidecar_path: str | Path) -> None:
        """Load HED definitions from a BIDS events.json sidecar.

        Looks for definitions in both the standard 'value.HED' field
        and the 'hed_defs.HED' field (used by PhysioNet MI).

        Args:
            sidecar_path: Path to a BIDS task-level events.json file.
        """
        sidecar_path = Path(sidecar_path)
        with open(sidecar_path) as f:
            sidecar = json.load(f)

        # Look for definitions in hed_defs (PhysioNet MI convention)
        hed_defs = sidecar.get("hed_defs", {}).get("HED", {})
        for def_string in hed_defs.values():
            hed_str = HedString(def_string, self._schema)
            self._def_dict.check_for_definitions(hed_str)

        # Also scan value.HED for inline definitions
        value_hed = sidecar.get("value", {}).get("HED", {})
        for hed_string in value_hed.values():
            hed_str = HedString(hed_string, self._schema)
            self._def_dict.check_for_definitions(hed_str)

        n_defs = len(self._def_dict.defs) if hasattr(self._def_dict, "defs") else 0
        logger.info(
            "Loaded definitions from %s (%d total definitions)",
            sidecar_path.name,
            n_defs,
        )

    def resolve_and_expand(self, hed_string: str) -> list[str]:
        """Resolve definitions and expand a HED string to long-form tags.

        Args:
            hed_string: A HED annotation string, possibly containing Def/ refs.

        Returns:
            List of unique long-form tag strings (no Def-expand tags).
        """
        hs = HedString(hed_string, self._schema, self._def_dict)
        hs.expand_defs()

        # Use long_tag attribute directly for full hierarchy paths
        all_tags = hs.get_all_tags()

        tags = set()
        for tag in all_tags:
            long = tag.long_tag
            # Skip Def-expand and Definition wrapper tags
            if "Def-expand" in long or "Definition" in long:
                continue
            # Skip Def references that weren't expanded
            if long.startswith("Property/Organizational-property/Def/"):
                continue
            tags.add(long)

        return sorted(tags)

    def _get_ancestors(self, long_form_tag: str) -> list[str]:
        """Get all ancestor tags for a long-form tag.

        For "Event/Agent-action" returns ["Event", "Event/Agent-action"].
        For "Action/Perform/Rest" returns ["Action", "Action/Perform", "Action/Perform/Rest"].
        """
        parts = long_form_tag.split("/")
        ancestors = []
        for i in range(1, len(parts) + 1):
            ancestors.append("/".join(parts[:i]))
        return ancestors

    @staticmethod
    def _tag_matches_branch_filter(tag: str, branch_filter: list[str]) -> bool:
        """Check if a tag belongs to any of the specified branch prefixes.

        Returns True if the tag exactly matches a prefix or is a descendant
        (starts with prefix + "/").
        """
        for branch in branch_filter:
            if tag == branch or tag.startswith(branch + "/"):
                return True
        return False

    def filtered_copy(self, branch_filter: list[str]) -> HEDVectorizer:
        """Create a new vectorizer containing only tags matching branch prefixes.

        Used to derive a process-HED or branch-restricted vectorizer from a
        saved full-vocabulary vectorizer (e.g., 1124-tag HEDit vocab).
        Ancestors of matching tags are included for hierarchy well-formedness.

        Args:
            branch_filter: List of branch prefixes to keep.

        Returns:
            A new HEDVectorizer with filtered vocabulary.
        """
        # Collect matching tags + their ancestors (if in source vocab)
        filtered_tags: set[str] = set()
        for tag in self._tag_to_idx:
            if self._tag_matches_branch_filter(tag, branch_filter):
                filtered_tags.add(tag)
                for ancestor in self._get_ancestors(tag):
                    if ancestor in self._tag_to_idx:
                        filtered_tags.add(ancestor)

        if not filtered_tags:
            raise ValueError(
                f"No tags match branch_filter={branch_filter} in vocab of "
                f"{self.vocab_size} tags."
            )

        # Build a new vectorizer with filtered vocab
        new_vec = HEDVectorizer.__new__(HEDVectorizer)
        new_vec._schema = self._schema
        new_vec._def_dict = self._def_dict
        new_vec._max_tag_depth = self._max_tag_depth
        new_vec._branch_filter = branch_filter

        sorted_tags = sorted(filtered_tags)
        new_vec._tag_to_idx = {tag: i for i, tag in enumerate(sorted_tags)}
        new_vec._idx_to_tag = {i: tag for tag, i in new_vec._tag_to_idx.items()}
        new_vec._tag_depths = {tag: tag.count("/") for tag in sorted_tags}
        new_vec._n_docs = self._n_docs
        new_vec._tag_doc_freq = {t: self._tag_doc_freq.get(t, 0) for t in sorted_tags}

        logger.info(
            "Filtered vectorizer: %d/%d tags (branch_filter=%s)",
            len(sorted_tags),
            self.vocab_size,
            branch_filter,
        )
        return new_vec

    def build_collapse_map(self, source_tag_to_idx: dict[str, int]) -> torch.Tensor:
        """Build a mapping matrix from a source vocab to this (collapsed) vocab.

        Used when pre-computed HED vectors are in the source vocab but training
        uses a collapsed vocab. Apply as: collapsed = (map @ source > 0).float()

        Args:
            source_tag_to_idx: Tag-to-index mapping from the source vocab
                (e.g., loaded from the saved vectorizer).

        Returns:
            (collapsed_vocab_size, source_vocab_size) binary matrix where
            entry [i, j] = 1 if source tag j maps to collapsed tag i.
        """
        mapping = torch.zeros(self.vocab_size, len(source_tag_to_idx))
        for src_tag, src_idx in source_tag_to_idx.items():
            collapsed = self._collapse_tag(src_tag)
            # The collapsed tag might itself not be in our vocab (if it was
            # pruned by min_frequency). Check ancestors too.
            for ancestor in self._get_ancestors(collapsed):
                tgt_idx = self._tag_to_idx.get(ancestor)
                if tgt_idx is not None:
                    mapping[tgt_idx, src_idx] = 1.0
        return mapping

    def _collapse_tag(self, long_form_tag: str) -> str:
        """Collapse a deep tag to its ancestor at max_tag_depth.

        If max_tag_depth is 0 (disabled) or the tag is already shallow enough,
        returns the tag unchanged. Otherwise returns the ancestor at max_tag_depth.
        """
        if self._max_tag_depth <= 0:
            return long_form_tag
        parts = long_form_tag.split("/")
        if len(parts) <= self._max_tag_depth + 1:
            return long_form_tag
        return "/".join(parts[: self._max_tag_depth + 1])

    def build_vocabulary(
        self,
        hed_strings: list[str],
        min_frequency: float = 0.0,
        branch_filter: list[str] | None = None,
    ) -> None:
        """Build the tag vocabulary from a collection of HED annotation strings.

        Resolves definitions, expands to long form, includes ancestors,
        and assigns indices. Tracks per-tag document frequency for IDF
        and pos_weight computation.

        Args:
            hed_strings: All HED annotation strings from all datasets.
            min_frequency: Minimum document frequency fraction (0.0 to 1.0)
                to include a tag. Tags appearing in fewer than
                min_frequency * N documents are dropped. Default 0.0
                (keep all).
            branch_filter: If set, restrict vocabulary to tags under these
                branch prefixes (plus their ancestors for hierarchy
                well-formedness). Used for process-HED mode.
        """
        if not 0.0 <= min_frequency <= 1.0:
            raise ValueError(
                f"min_frequency must be between 0.0 and 1.0, got {min_frequency}"
            )
        all_tags: set[str] = set()
        doc_freq: dict[str, int] = {}
        parse_failures = 0
        n_docs = 0

        for hed_string in hed_strings:
            try:
                tags = self.resolve_and_expand(hed_string)
                doc_tags: set[str] = set()
                for tag in tags:
                    # Collapse deep tags before ancestor expansion
                    collapsed = self._collapse_tag(tag)
                    for ancestor in self._get_ancestors(collapsed):
                        all_tags.add(ancestor)
                        doc_tags.add(ancestor)
                for t in doc_tags:
                    doc_freq[t] = doc_freq.get(t, 0) + 1
                n_docs += 1
            except (ValueError, KeyError, AttributeError) as e:
                parse_failures += 1
                logger.warning(
                    "Failed to parse HED string '%s': %s", hed_string[:80], e
                )
                continue

        if parse_failures > 0:
            logger.warning(
                "HED parsing: %d/%d strings failed",
                parse_failures,
                len(hed_strings),
            )
        if parse_failures == len(hed_strings) and len(hed_strings) > 0:
            raise ValueError(
                f"All {len(hed_strings)} HED strings failed to parse. "
                "Check schema version compatibility and sidecar format."
            )
        if not all_tags:
            raise ValueError(
                "No valid HED tags found after parsing. Cannot build vocabulary."
            )

        # Filter by minimum frequency
        if min_frequency > 0.0 and n_docs > 0:
            min_count = min_frequency * n_docs
            before = len(all_tags)
            all_tags = {t for t in all_tags if doc_freq.get(t, 0) >= min_count}
            dropped = before - len(all_tags)
            if dropped > 0:
                logger.info(
                    "Dropped %d tags below min_frequency=%.3f (%d docs)",
                    dropped,
                    min_frequency,
                    int(min_count),
                )
            if not all_tags:
                raise ValueError(
                    f"All tags dropped by min_frequency={min_frequency}. "
                    "Lower the threshold or add more data."
                )

        # Filter by branch prefixes (process-HED mode)
        if branch_filter:
            before = len(all_tags)
            # Keep tags matching any branch prefix + their ancestors
            filtered: set[str] = set()
            for tag in all_tags:
                if self._tag_matches_branch_filter(tag, branch_filter):
                    filtered.add(tag)
                    # Include ancestors for hierarchy well-formedness
                    for ancestor in self._get_ancestors(tag):
                        filtered.add(ancestor)
            all_tags = filtered
            logger.info(
                "Branch filter kept %d/%d tags (prefixes: %s)",
                len(all_tags),
                before,
                branch_filter,
            )
            if not all_tags:
                raise ValueError(
                    f"All tags dropped by branch_filter={branch_filter}. "
                    "Check that HED sidecars contain process-level tags."
                )
        self._branch_filter = branch_filter or []

        # Sort for deterministic ordering
        sorted_tags = sorted(all_tags)
        self._tag_to_idx = {tag: i for i, tag in enumerate(sorted_tags)}
        self._idx_to_tag = {i: tag for tag, i in self._tag_to_idx.items()}

        # Store document frequencies for IDF/pos_weight
        self._n_docs = n_docs
        self._tag_doc_freq = {t: doc_freq.get(t, 0) for t in sorted_tags}

        # Compute depths (number of / separators)
        self._tag_depths = {}
        for tag in sorted_tags:
            self._tag_depths[tag] = tag.count("/")  # root = 0, first level = 1, etc.

        logger.info(
            "Built vocabulary with %d tags (max depth %d, N=%d docs)",
            len(sorted_tags),
            max(self._tag_depths.values()) if self._tag_depths else 0,
            n_docs,
        )

    def vectorize(self, hed_string: str) -> torch.Tensor:
        """Convert a HED annotation string to a multi-hot binary vector.

        Resolves definitions, expands to long form, sets bits for all
        tags and their ancestors.

        Args:
            hed_string: A HED annotation string.

        Returns:
            (vocab_size,) float tensor with 1.0 for active tags, 0.0 otherwise.

        Raises:
            RuntimeError: if vocabulary has not been built yet.
        """
        if not self._tag_to_idx:
            raise RuntimeError("Vocabulary not built. Call build_vocabulary() first.")

        vector = torch.zeros(self.vocab_size, dtype=torch.float32)

        try:
            tags = self.resolve_and_expand(hed_string)
        except (ValueError, KeyError, AttributeError) as e:
            logger.error(
                "Failed to vectorize HED string '%s': %s. "
                "Returning zero vector; this will corrupt training labels.",
                hed_string[:80],
                e,
            )
            return vector

        for tag in tags:
            collapsed = self._collapse_tag(tag)
            for ancestor in self._get_ancestors(collapsed):
                idx = self._tag_to_idx.get(ancestor)
                if idx is not None:
                    vector[idx] = 1.0

        return vector

    # Semantic level branch prefixes (Hypothesis A: Visual Processing Pipeline).
    # Each tag is assigned to a level based on its longest matching prefix.
    # Within each level, depth relative to the branch root applies alpha decay.
    _SEMANTIC_LEVELS: dict[int, list[str]] = {
        # Level 0: Meta/Boilerplate (excluded from loss)
        # Includes root-level tags (Event, Item, Property) that appear
        # in nearly every annotation due to ancestor inclusion.
        0: [
            "Event",
            "Event/Sensory-event",
            "Event/Experiment-control",
            "Event/Experiment-procedure",
            "Event/Experiment-structure",
            "Item",
            "Property",
            "Property/Sensory-property",
            "Property/Sensory-property/Sensory-presentation",
            "Property/Task-property",
            "Property/Task-property/Task-event-role",
            "Property/Task-property/Task-event-role/Experimental-stimulus",
            "Property/Organizational-property",
            "Property/Informational-property",
            "Property/Agent-property",
        ],
        # Level 1: Scene Category
        1: [
            "Property/Environmental-property",
            "Item/Object/Man-made-object/Building",
        ],
        # Level 2: Entity Identity
        2: [
            "Agent",
            "Item/Biological-item",
            "Item/Object",
        ],
        # Level 3: Action/State
        3: [
            "Action",
            "Property/Agent-property/Agent-state",
            "Event/Agent-action",
            "Event/Data-feature",
            "Event/Measurement-event",
            "Property/Task-property/Task-attentional-demand",
        ],
        # Level 4: Attributes & Relations
        4: [
            "Property/Sensory-property/Sensory-attribute",
            "Property/Data-property",
            "Property/Agent-property/Agent-trait",
            "Property/Agent-property/Agent-task-role",
            "Property/Task-property/Task-action-type",
            "Property/Task-property/Task-effect-evidence",
            "Property/Task-property/Task-stimulus-role",
            "Property/Task-property/Task-relationship",
            "Relation",
            "Item/Language-item",
            "Item/Sound",
            "Property/Data-property/Data-resolution",
        ],
    }

    _LEVEL_WEIGHTS: dict[int, float] = {
        0: 0.0,
        1: 1.0,
        2: 2.0,
        3: 2.0,
        4: 0.5,
    }

    def classify_tag(self, long_form_tag: str) -> tuple[int, int]:
        """Classify a long-form tag into its semantic level and branch depth.

        Uses longest prefix match against _SEMANTIC_LEVELS. Branch depth
        is the number of path components after the matched prefix.

        Args:
            long_form_tag: Full long-form HED tag path.

        Returns:
            (semantic_level, branch_depth) tuple. Defaults to (4, schema_depth)
            if no prefix matches (conservative: treat unknown tags as attributes).
        """
        best_level = 4  # default: attribute level
        best_prefix_len = 0
        best_branch_depth = long_form_tag.count("/")

        for level, prefixes in self._SEMANTIC_LEVELS.items():
            for prefix in prefixes:
                if (
                    long_form_tag == prefix or long_form_tag.startswith(prefix + "/")
                ) and len(prefix) > best_prefix_len:
                    best_level = level
                    best_prefix_len = len(prefix)
                    # Branch depth = components after the prefix
                    remaining = long_form_tag[len(prefix) :]
                    best_branch_depth = remaining.count("/")

        return best_level, best_branch_depth

    def get_semantic_weights(self, alpha: float = 0.7) -> torch.Tensor:
        """Compute per-tag weights using semantic level + within-branch depth.

        Weight formula: level_weight(level) * alpha^branch_depth

        Level weights (Hypothesis A: Visual Processing Pipeline):
            Level 0 (Meta):      0.0  (excluded)
            Level 1 (Scene):     1.0
            Level 2 (Entity):    2.0
            Level 3 (Action):    2.0
            Level 4 (Attribute): 0.5

        Args:
            alpha: Within-branch depth decay factor. Default 0.7.

        Returns:
            (vocab_size,) float tensor of per-tag weights.
        """
        if not self._tag_to_idx:
            raise RuntimeError("Vocabulary not built. Call build_vocabulary() first.")

        weights = torch.zeros(self.vocab_size, dtype=torch.float32)
        for tag, idx in self._tag_to_idx.items():
            level, branch_depth = self.classify_tag(tag)
            level_weight = self._LEVEL_WEIGHTS.get(level, 0.5)
            weights[idx] = level_weight * (alpha**branch_depth)

        return weights

    def get_depth_weights(self, alpha: float = 0.7) -> torch.Tensor:
        """Compute per-tag depth weights for the depth-weighted BCE loss.

        Weight formula: w(depth) = alpha^depth
        Root tags (depth 0) get weight 1.0; deeper tags get progressively less.

        Args:
            alpha: Decay factor per depth level. Default 0.7 gives:
                depth 0: 1.0, depth 1: 0.7, depth 2: 0.49, depth 3: 0.34

        Returns:
            (vocab_size,) float tensor of per-tag weights.

        Raises:
            RuntimeError: if vocabulary has not been built yet.
        """
        if not self._tag_to_idx:
            raise RuntimeError("Vocabulary not built. Call build_vocabulary() first.")

        weights = torch.zeros(self.vocab_size, dtype=torch.float32)
        for tag, idx in self._tag_to_idx.items():
            depth = self._tag_depths[tag]
            weights[idx] = alpha**depth

        return weights

    def get_idf_weights(self) -> torch.Tensor:
        """Compute IDF weights from document frequencies.

        IDF = log(1 + N / (df + 1)) which is guaranteed non-negative.
        Tags appearing in every document get a small positive weight;
        rare tags get higher weight. This avoids the negative-weight
        problem of the classic log(N / (df + 1)) formula.

        Returns:
            (vocab_size,) float tensor of non-negative IDF weights.

        Raises:
            RuntimeError: if vocabulary has not been built yet.
        """
        if not self._tag_to_idx:
            raise RuntimeError("Vocabulary not built. Call build_vocabulary() first.")
        if self._n_docs == 0:
            raise RuntimeError(
                "No document frequency data. Rebuild vocabulary with build_vocabulary()."
            )

        weights = torch.zeros(self.vocab_size, dtype=torch.float32)
        for tag, idx in self._tag_to_idx.items():
            df = self._tag_doc_freq.get(tag, 0)
            weights[idx] = math.log(1 + self._n_docs / (df + 1))

        return weights

    def get_pos_weights(self, max_pos_weight: float = 10.0) -> torch.Tensor:
        """Compute pos_weight for BCEWithLogitsLoss class imbalance.

        pos_weight[i] = (N - n_positive[i]) / n_positive[i] for each tag,
        where N = total documents and n_positive = document frequency. Tags
        with zero document frequency get pos_weight = 1.0 to avoid division
        by zero.

        Unclamped weights for rare tags (e.g., appearing in 0.1% of docs) can
        reach ~1000, which causes gradient instability. The max_pos_weight cap
        prevents extreme values while preserving the relative ordering.

        Args:
            max_pos_weight: Maximum allowed pos_weight value. Tags rarer than
                1/(max_pos_weight+1) of the corpus are clamped to this value.
                Default 10.0 keeps gradients stable while correcting imbalance.

        Returns:
            (vocab_size,) float tensor of pos_weights clamped to [0, max_pos_weight].

        Raises:
            ValueError: if max_pos_weight is not positive.
            RuntimeError: if vocabulary has not been built yet.
        """
        if max_pos_weight <= 0:
            raise ValueError(f"max_pos_weight must be positive, got {max_pos_weight}")
        if not self._tag_to_idx:
            raise RuntimeError("Vocabulary not built. Call build_vocabulary() first.")
        if self._n_docs == 0:
            raise RuntimeError(
                "No document frequency data. Rebuild vocabulary with build_vocabulary()."
            )

        weights = torch.ones(self.vocab_size, dtype=torch.float32)
        for tag, idx in self._tag_to_idx.items():
            n_pos = self._tag_doc_freq.get(tag, 0)
            if n_pos > 0:
                weights[idx] = min((self._n_docs - n_pos) / n_pos, max_pos_weight)

        return weights

    def get_descendants_matrix(self) -> torch.Tensor:
        """Return (vocab, vocab) bool matrix of strict ancestor->descendant.

        ``M[i, j] = True`` iff ``tag_j`` is a strict descendant of ``tag_i``
        (i.e. ``idx_to_tag[j].startswith(idx_to_tag[i] + "/")``). Used by
        loss flavors that strip ancestor tags from the multi-hot HED
        target at training time (preprocessed perlevel / top-k MI). Built
        once per vectorizer; size 1124² ≈ 1.26 MB at vocab_size=1124.
        """
        if not self._tag_to_idx:
            raise RuntimeError("Vocabulary not built. Call build_vocabulary() first.")
        v = self.vocab_size
        tags = [self._idx_to_tag[i] for i in range(v)]
        m = torch.zeros(v, v, dtype=torch.bool)
        for i, ti in enumerate(tags):
            prefix = ti + "/"
            for j, tj in enumerate(tags):
                if j != i and tj.startswith(prefix):
                    m[i, j] = True
        return m

    def get_level_partition(self) -> dict[int, list[int]]:
        """Return mapping {level -> list of tag indices} using Hypothesis A.

        Levels 1..4 (level 0 is meta and excluded). Tags are assigned via
        :meth:`classify_tag` (longest-prefix match against
        :attr:`_SEMANTIC_LEVELS`), so every vocabulary tag lands in exactly
        one bucket. Used by preprocessed PerLevelSoftmaxHEDLoss.
        """
        if not self._tag_to_idx:
            raise RuntimeError("Vocabulary not built. Call build_vocabulary() first.")
        partition: dict[int, list[int]] = {1: [], 2: [], 3: [], 4: []}
        for tag, idx in self._tag_to_idx.items():
            level, _ = self.classify_tag(tag)
            if level == 0:
                continue
            partition.setdefault(level, []).append(idx)
        return partition

    def get_combined_weights(self, alpha: float = 0.7) -> torch.Tensor:
        """Compute combined semantic + IDF weights.

        total_weight = level_weight * alpha^branch_depth * idf_weight

        This multiplies the semantic level weights (which encode neuroscience
        priors) with IDF weights (which encode corpus statistics), giving
        the best of both: domain-informed weighting with data-driven
        rare-tag upweighting.

        Args:
            alpha: Within-branch depth decay factor. Default 0.7.

        Returns:
            (vocab_size,) float tensor of combined weights.
        """
        return self.get_semantic_weights(alpha=alpha) * self.get_idf_weights()

    def get_hierarchy_init_embeddings(
        self,
        embed_dim: int,
        decay: float = 0.5,
        noise_scale: float = 0.02,
    ) -> torch.Tensor:
        """Generate hierarchy-aware initial embeddings for tag embedding head.

        Root tags (depth 0) get random embeddings scaled by noise_scale.
        Children inherit their parent's embedding plus Gaussian noise
        scaled by noise_scale * decay^depth. This encodes the HED tree
        structure: siblings start close, cousins further apart.

        Args:
            embed_dim: Embedding dimension (must match model embed_dim).
            decay: Per-depth noise decay factor. Deeper children get
                smaller perturbations from their parent. Default 0.5.
            noise_scale: Base scale for random initialization. Default 0.02
                (matches typical transformer init).

        Returns:
            (vocab_size, embed_dim) float tensor of initial tag embeddings.

        Raises:
            RuntimeError: if vocabulary has not been built yet.
        """
        if not self._tag_to_idx:
            raise RuntimeError("Vocabulary not built. Call build_vocabulary() first.")

        embeddings = torch.zeros(self.vocab_size, embed_dim)

        # Process tags in depth order so parents are initialized before children
        tags_by_depth: dict[int, list[str]] = {}
        for tag, depth in self._tag_depths.items():
            tags_by_depth.setdefault(depth, []).append(tag)

        n_orphans = 0
        for depth in sorted(tags_by_depth):
            for tag in tags_by_depth[depth]:
                idx = self._tag_to_idx[tag]
                if depth == 0:
                    # Root tags: random initialization
                    embeddings[idx] = torch.randn(embed_dim) * noise_scale
                else:
                    # Find parent: tag with one fewer path component
                    parent_tag = "/".join(tag.split("/")[:-1])
                    parent_idx = self._tag_to_idx.get(parent_tag)
                    if parent_idx is not None:
                        perturbation = (
                            torch.randn(embed_dim) * noise_scale * (decay**depth)
                        )
                        embeddings[idx] = embeddings[parent_idx] + perturbation
                    else:
                        # Parent not in vocab (pruned by min_frequency); init as root
                        embeddings[idx] = torch.randn(embed_dim) * noise_scale
                        n_orphans += 1

        if n_orphans > 0:
            pct = n_orphans / self.vocab_size * 100
            logger.info(
                "Hierarchy init: %d/%d tags (%.1f%%) fell back to root init "
                "(parent pruned from vocab)",
                n_orphans,
                self.vocab_size,
                pct,
            )
            if pct > 25:
                logger.warning(
                    "Over 25%% of tags lost hierarchy structure in init. "
                    "Consider lowering min_frequency to preserve parent tags."
                )

        return embeddings

    def save(self, path: str | Path) -> None:
        """Save vectorizer state for reproducibility."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "tag_to_idx": self._tag_to_idx,
                "idx_to_tag": self._idx_to_tag,
                "tag_depths": self._tag_depths,
                "tag_doc_freq": self._tag_doc_freq,
                "n_docs": self._n_docs,
                "max_tag_depth": self._max_tag_depth,
                "branch_filter": getattr(self, "_branch_filter", []),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load a saved vectorizer state."""
        data = torch.load(Path(path), weights_only=True)
        self._tag_to_idx = data["tag_to_idx"]
        self._idx_to_tag = data["idx_to_tag"]
        self._tag_depths = data["tag_depths"]
        self._tag_doc_freq = data.get("tag_doc_freq", {})
        self._n_docs = data.get("n_docs", 0)
        self._max_tag_depth = data.get("max_tag_depth", 0)
        self._branch_filter: list[str] = data.get("branch_filter", [])

    @classmethod
    def from_sidecars(
        cls,
        sidecar_paths: list[str | Path] | list[Path],
        schema_version: str = "8.3.0",
        min_frequency: float = 0.0,
        max_tag_depth: int = 0,
        branch_filter: list[str] | None = None,
    ) -> HEDVectorizer:
        """Build a vectorizer from a list of BIDS events.json sidecars.

        Loads definitions from all sidecars, extracts all HED strings,
        and builds the shared vocabulary.

        Args:
            sidecar_paths: Paths to BIDS task-level events.json files.
            schema_version: HED schema version to use.
            min_frequency: Minimum document frequency fraction (0.0 to 1.0)
                to include a tag. Tags appearing in fewer than
                min_frequency * N documents are dropped. Default 0.0
                (keep all).
            max_tag_depth: Collapse tags deeper than this to ancestor.
            branch_filter: If set, restrict vocabulary to these branch
                prefixes. For process-HED, use PROCESS_BRANCHES.

        Returns:
            A ready-to-use HEDVectorizer.
        """
        vectorizer = cls(schema_version=schema_version, max_tag_depth=max_tag_depth)

        all_hed_strings = []
        for sidecar_path in sidecar_paths:
            sidecar_path = Path(sidecar_path)
            vectorizer.load_definitions_from_sidecar(sidecar_path)

            with open(sidecar_path) as f:
                sidecar = json.load(f)

            value_hed = sidecar.get("value", {}).get("HED", {})
            all_hed_strings.extend(value_hed.values())

        vectorizer.build_vocabulary(
            all_hed_strings,
            min_frequency=min_frequency,
            branch_filter=branch_filter,
        )
        return vectorizer

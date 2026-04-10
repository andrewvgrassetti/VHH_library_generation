import itertools
import logging
import random
import math
import pandas as pd
from vhh_library.humanness import HumAnnotator
from vhh_library.stability import StabilityScorer
from vhh_library.developability import (
    PTMLiabilityScorer,
    ClearanceRiskScorer,
    SurfaceHydrophobicityScorer,
)
from vhh_library.sequence import VHHSequence

logger = logging.getLogger(__name__)

# When the combinatorial space exceeds this number of total combinations we
# switch from exhaustive enumeration to random sampling.
_SAMPLING_THRESHOLD = 50_000


def _total_combinations(n: int, k_min: int, k_max: int) -> int:
    """Return sum of C(n, k) for k in [k_min, k_max], capped to avoid overflow."""
    total = 0
    for k in range(k_min, k_max + 1):
        try:
            total += math.comb(n, k)
        except (ValueError, OverflowError):
            return _SAMPLING_THRESHOLD + 1  # signal "too large"
        if total > _SAMPLING_THRESHOLD * 100:
            return total  # early exit
    return total


class MutationEngine:
    """Generate and rank VHH variant libraries.

    Supports up to five scoring metrics.  Each metric has a boolean *enabled*
    flag and a *weight*.  Disabled metrics are excluded from the combined
    score but their raw values are still recorded in the output DataFrame.

    Parameters
    ----------
    humanness_scorer, stability_scorer:
        Always-required core scorers.
    ptm_scorer, clearance_scorer, hydrophobicity_scorer:
        Optional developability scorers.  If ``None`` they are created
        lazily the first time a metric is enabled.
    weights:
        Dict mapping metric name → float weight.  Unknown keys are ignored.
    enabled_metrics:
        Dict mapping metric name → bool.  ``True`` means the metric
        participates in the combined score.  Defaults to humanness +
        stability enabled.
    """

    METRIC_NAMES = (
        "humanness",
        "stability",
        "ptm_liability",
        "clearance_risk",
        "surface_hydrophobicity",
    )

    def __init__(
        self,
        humanness_scorer: HumAnnotator,
        stability_scorer: StabilityScorer,
        *,
        ptm_scorer: PTMLiabilityScorer | None = None,
        clearance_scorer: ClearanceRiskScorer | None = None,
        hydrophobicity_scorer: SurfaceHydrophobicityScorer | None = None,
        w_humanness: float = 0.6,
        w_stability: float = 0.4,
        weights: dict | None = None,
        enabled_metrics: dict | None = None,
    ):
        self.humanness_scorer = humanness_scorer
        self.stability_scorer = stability_scorer
        self._ptm_scorer = ptm_scorer
        self._clearance_scorer = clearance_scorer
        self._hydrophobicity_scorer = hydrophobicity_scorer

        # Legacy two-weight API
        self.w_humanness = w_humanness
        self.w_stability = w_stability

        # Multi-metric weights (override legacy when provided)
        self.weights: dict[str, float] = {
            "humanness": w_humanness,
            "stability": w_stability,
            "ptm_liability": 0.0,
            "clearance_risk": 0.0,
            "surface_hydrophobicity": 0.0,
        }
        if weights:
            for k, v in weights.items():
                if k in self.weights:
                    self.weights[k] = v

        # Which metrics participate in combined score
        self.enabled_metrics: dict[str, bool] = {
            "humanness": True,
            "stability": True,
            "ptm_liability": False,
            "clearance_risk": False,
            "surface_hydrophobicity": False,
        }
        if enabled_metrics:
            for k, v in enabled_metrics.items():
                if k in self.enabled_metrics:
                    self.enabled_metrics[k] = v

    # -- lazy scorer accessors ------------------------------------------------

    @property
    def ptm_scorer(self) -> PTMLiabilityScorer:
        if self._ptm_scorer is None:
            self._ptm_scorer = PTMLiabilityScorer()
        return self._ptm_scorer

    @property
    def clearance_scorer(self) -> ClearanceRiskScorer:
        if self._clearance_scorer is None:
            self._clearance_scorer = ClearanceRiskScorer()
        return self._clearance_scorer

    @property
    def hydrophobicity_scorer(self) -> SurfaceHydrophobicityScorer:
        if self._hydrophobicity_scorer is None:
            self._hydrophobicity_scorer = SurfaceHydrophobicityScorer()
        return self._hydrophobicity_scorer

    # -- scoring helpers ------------------------------------------------------

    def _active_weights(self) -> dict[str, float]:
        """Return normalised weights for enabled metrics only."""
        raw = {k: self.weights[k] for k in self.METRIC_NAMES if self.enabled_metrics.get(k)}
        total = sum(raw.values())
        if total == 0:
            return raw
        return {k: v / total for k, v in raw.items()}

    def _score_variant(self, vhh: VHHSequence) -> dict[str, float]:
        """Score a variant across all five metrics.  Returns raw scores."""
        scores: dict[str, float] = {}
        scores["humanness"] = self.humanness_scorer.score(vhh)["composite_score"]
        scores["stability"] = self.stability_scorer.score(vhh)["composite_score"]
        scores["ptm_liability"] = self.ptm_scorer.score(vhh)["composite_score"]
        scores["clearance_risk"] = self.clearance_scorer.score(vhh)["composite_score"]
        scores["surface_hydrophobicity"] = self.hydrophobicity_scorer.score(vhh)["composite_score"]
        return scores

    def _combined_score(self, raw_scores: dict[str, float]) -> float:
        aw = self._active_weights()
        return sum(aw.get(k, 0.0) * raw_scores[k] for k in self.METRIC_NAMES)

    # -- public API -----------------------------------------------------------

    def rank_single_mutations(self, vhh_sequence: VHHSequence, off_limits: set = None,
                              forbidden_substitutions: dict | None = None) -> pd.DataFrame:
        if off_limits is None:
            off_limits = set()
        if forbidden_substitutions is None:
            forbidden_substitutions = {}
        suggestions = self.humanness_scorer.get_mutation_suggestions(
            vhh_sequence, off_limits, forbidden_substitutions=forbidden_substitutions
        )

        active_w = self._active_weights()
        rows = []
        for s in suggestions:
            imgt_pos = s["position"]
            delta_h = s["delta_humanness"]
            delta_s = self.stability_scorer.predict_mutation_effect(vhh_sequence, imgt_pos, s["suggested_aa"])

            deltas: dict[str, float] = {"humanness": delta_h, "stability": delta_s}

            # Optional metric deltas (only computed when enabled to save time)
            if self.enabled_metrics.get("ptm_liability"):
                deltas["ptm_liability"] = self.ptm_scorer.predict_mutation_effect(
                    vhh_sequence, imgt_pos, s["suggested_aa"])
            else:
                deltas["ptm_liability"] = 0.0

            if self.enabled_metrics.get("clearance_risk"):
                deltas["clearance_risk"] = self.clearance_scorer.predict_mutation_effect(
                    vhh_sequence, imgt_pos, s["suggested_aa"])
            else:
                deltas["clearance_risk"] = 0.0

            if self.enabled_metrics.get("surface_hydrophobicity"):
                deltas["surface_hydrophobicity"] = self.hydrophobicity_scorer.predict_mutation_effect(
                    vhh_sequence, imgt_pos, s["suggested_aa"])
            else:
                deltas["surface_hydrophobicity"] = 0.0

            combined = sum(active_w.get(k, 0.0) * deltas[k] for k in self.METRIC_NAMES)

            row = {
                "position": imgt_pos,
                "imgt_pos": imgt_pos,
                "original_aa": s["original_aa"],
                "suggested_aa": s["suggested_aa"],
                "delta_humanness": round(delta_h, 4),
                "delta_stability": round(delta_s, 4),
                "delta_ptm_liability": round(deltas["ptm_liability"], 4),
                "delta_clearance_risk": round(deltas["clearance_risk"], 4),
                "delta_surface_hydrophobicity": round(deltas["surface_hydrophobicity"], 4),
                "combined_score": round(combined, 4),
                "reason": s["reason"],
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
        return df

    def apply_mutations(self, sequence: str, mutations: list) -> str:
        seq_list = list(sequence)
        for idx, new_aa in mutations:
            if 0 <= idx < len(seq_list):
                seq_list[idx] = new_aa
        return "".join(seq_list)

    # -- library generation ---------------------------------------------------

    def generate_library(self, vhh_sequence: VHHSequence, top_mutations: pd.DataFrame,
                         n_mutations: int, max_variants: int = 10000,
                         min_mutations: int = 1) -> pd.DataFrame:
        """Generate a combinatorial variant library.

        Uses random sampling when the combinatorial space is very large
        (controlled by ``_SAMPLING_THRESHOLD``), ensuring the function
        completes in seconds rather than hours.

        Parameters
        ----------
        vhh_sequence:
            The parent VHH sequence.
        top_mutations:
            Ranked single-mutation DataFrame from ``rank_single_mutations``.
        n_mutations:
            Maximum number of mutations per variant.
        max_variants:
            Upper limit on total variants generated.
        min_mutations:
            Minimum number of mutations per variant (default 1).
        """
        if len(top_mutations) == 0:
            return pd.DataFrame()

        orig_min = min_mutations
        min_mutations = max(1, min(min_mutations, n_mutations))
        if min_mutations != orig_min:
            logger.warning(
                "min_mutations=%d was clamped to %d (valid range: 1–%d)",
                orig_min, min_mutations, n_mutations,
            )

        candidates = top_mutations.head(min(len(top_mutations), n_mutations * 3))
        mut_list = list(candidates.itertuples(index=False))

        n_cands = len(mut_list)
        total_combos = _total_combinations(n_cands, min_mutations, n_mutations)

        if total_combos <= _SAMPLING_THRESHOLD:
            # Exhaustive enumeration (fast enough)
            rows = self._generate_exhaustive(
                vhh_sequence, mut_list, n_mutations, min_mutations,
                max_variants,
            )
        else:
            # Random sampling for large spaces
            logger.info(
                "Combinatorial space (~%.2e) exceeds threshold; using random sampling.",
                total_combos,
            )
            rows = self._generate_sampled(
                vhh_sequence, mut_list, n_mutations, min_mutations,
                max_variants,
            )

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
        return df

    # -- private generation strategies ----------------------------------------

    def _build_variant_row(self, vhh_sequence, selected, variant_counter):
        """Score a combination and return a row dict (or ``None`` if invalid)."""
        positions = [m.imgt_pos for m in selected]
        if len(set(positions)) != len(positions):
            return None

        muts_0idx = [(m.imgt_pos - 1, m.suggested_aa) for m in selected]
        new_seq = self.apply_mutations(vhh_sequence.sequence, muts_0idx)
        mut_str = ", ".join(f"{m.original_aa}{m.imgt_pos}{m.suggested_aa}" for m in selected)

        mutant_vhh = VHHSequence(new_seq)
        raw_scores = self._score_variant(mutant_vhh)
        combined = self._combined_score(raw_scores)

        return {
            "variant_id": f"VAR-{variant_counter + 1:06d}",
            "mutations": mut_str,
            "n_mutations": len(selected),
            "humanness_score": round(raw_scores["humanness"], 4),
            "stability_score": round(raw_scores["stability"], 4),
            "ptm_liability_score": round(raw_scores["ptm_liability"], 4),
            "clearance_risk_score": round(raw_scores["clearance_risk"], 4),
            "surface_hydrophobicity_score": round(raw_scores["surface_hydrophobicity"], 4),
            "combined_score": round(combined, 4),
            "aa_sequence": new_seq,
        }

    def _generate_exhaustive(self, vhh_sequence, mut_list, n_mutations,
                             min_mutations, max_variants):
        rows = []
        variant_counter = 0
        for k in range(min_mutations, n_mutations + 1):
            for combo in itertools.combinations(range(len(mut_list)), k):
                if variant_counter >= max_variants:
                    break
                selected = [mut_list[i] for i in combo]
                row = self._build_variant_row(vhh_sequence, selected, variant_counter)
                if row is not None:
                    rows.append(row)
                    variant_counter += 1
            if variant_counter >= max_variants:
                break
        return rows

    def _generate_sampled(self, vhh_sequence, mut_list, n_mutations,
                          min_mutations, max_variants):
        """Generate variants by random sampling of position-compatible combos.

        Strategy: for each variant we pick a random k in [min_mutations,
        n_mutations], then greedily sample *positions* (ensuring no duplicates)
        and randomly select one mutation per chosen position.  This avoids
        generating combinations that will be discarded due to position
        conflicts, and completes in O(max_variants) time.
        """
        # Build position → list of candidate mutations
        pos_muts: dict[int, list] = {}
        for m in mut_list:
            pos_muts.setdefault(m.imgt_pos, []).append(m)
        unique_positions = list(pos_muts.keys())

        if len(unique_positions) < min_mutations:
            return []

        rows = []
        seen_combos: set[frozenset] = set()
        variant_counter = 0
        # Allow generous number of attempts before giving up
        max_attempts = max_variants * 10

        for _ in range(max_attempts):
            if variant_counter >= max_variants:
                break

            k = random.randint(min_mutations, min(n_mutations, len(unique_positions)))
            chosen_positions = random.sample(unique_positions, k)

            # Pick one mutation per position
            selected = []
            combo_key_parts = []
            for pos in sorted(chosen_positions):
                m = random.choice(pos_muts[pos])
                selected.append(m)
                combo_key_parts.append((pos, m.suggested_aa))
            combo_key = frozenset(combo_key_parts)

            if combo_key in seen_combos:
                continue
            seen_combos.add(combo_key)

            row = self._build_variant_row(vhh_sequence, selected, variant_counter)
            if row is not None:
                rows.append(row)
                variant_counter += 1

        return rows

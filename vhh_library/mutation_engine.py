from __future__ import annotations

import itertools
import logging
import math
import random
from typing import Optional
import pandas as pd
from vhh_library.humanness import HumAnnotator
from vhh_library.stability import StabilityScorer
from vhh_library.developability import (
    PTMLiabilityScorer,
    ClearanceRiskScorer,
    SurfaceHydrophobicityScorer,
)
from vhh_library.orthogonal_scoring import (
    HumanStringContentScorer,
    ConsensusStabilityScorer,
)
from vhh_library.sequence import VHHSequence

logger = logging.getLogger(__name__)

# When the combinatorial space exceeds this threshold we switch from exhaustive
# enumeration to random sampling.
_SAMPLING_THRESHOLD = 50_000

# When the combinatorial space exceeds this threshold the "auto" strategy
# activates iterative refinement instead of pure random sampling.
_ITERATIVE_THRESHOLD = 1_000_000

# Minimum score improvement per top-K average to continue iterative refinement.
_CONVERGENCE_THRESHOLD = 1e-4

# Minimum average-score improvement required to unlock an anchor position.
_ANCHOR_UNLOCK_THRESHOLD = 0.01


def _parse_mut_str(mut_str: str) -> list[tuple[int, str]]:
    """Parse a mutations string like ``'X1Y, A2B'`` to ``[(1, 'Y'), (2, 'B')]``."""
    result = []
    for part in mut_str.split(","):
        part = part.strip()
        if len(part) >= 3:
            try:
                pos = int(part[1:-1])
                aa = part[-1]
                result.append((pos, aa))
            except (ValueError, IndexError):
                logger.debug("Could not parse mutation token %r; skipping.", part)
    return result


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
        hsc_scorer: HumanStringContentScorer | None = None,
        consensus_scorer: ConsensusStabilityScorer | None = None,
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
        self._hsc_scorer = hsc_scorer
        self._consensus_scorer = consensus_scorer

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

    @property
    def hsc_scorer(self) -> HumanStringContentScorer:
        if self._hsc_scorer is None:
            self._hsc_scorer = HumanStringContentScorer()
        return self._hsc_scorer

    @property
    def consensus_scorer(self) -> ConsensusStabilityScorer:
        if self._consensus_scorer is None:
            self._consensus_scorer = ConsensusStabilityScorer()
        return self._consensus_scorer

    # -- scoring helpers ------------------------------------------------------

    def _active_weights(self) -> dict[str, float]:
        """Return normalised weights for enabled metrics only."""
        raw = {k: self.weights[k] for k in self.METRIC_NAMES if self.enabled_metrics.get(k)}
        total = sum(raw.values())
        if total == 0:
            return raw
        return {k: v / total for k, v in raw.items()}

    def _score_variant(self, vhh: VHHSequence) -> dict[str, float]:
        """Score a variant across all metrics.  Returns raw scores.

        Includes the five primary metrics used for the combined score
        plus the two orthogonal scores (which are recorded but never
        participate in the combined score).
        """
        scores: dict[str, float] = {}
        scores["humanness"] = self.humanness_scorer.score(vhh)["composite_score"]
        stability_result = self.stability_scorer.score(vhh)
        scores["stability"] = stability_result["composite_score"]
        scores["aggregation_score"] = stability_result["aggregation_score"]
        scores["charge_balance_score"] = stability_result["charge_balance_score"]
        scores["hydrophobic_core_score"] = stability_result["hydrophobic_core_score"]
        scores["disulfide_score"] = stability_result["disulfide_score"]
        scores["vhh_hallmark_score"] = stability_result["vhh_hallmark_score"]
        scores["ptm_liability"] = self.ptm_scorer.score(vhh)["composite_score"]
        scores["clearance_risk"] = self.clearance_scorer.score(vhh)["composite_score"]
        scores["surface_hydrophobicity"] = self.hydrophobicity_scorer.score(vhh)["composite_score"]
        # Orthogonal scores (informational only – not in combined score)
        scores["orthogonal_humanness"] = self.hsc_scorer.score(vhh)["composite_score"]
        scores["orthogonal_stability"] = self.consensus_scorer.score(vhh)["composite_score"]
        return scores

    def _combined_score(self, raw_scores: dict[str, float]) -> float:
        aw = self._active_weights()
        return sum(aw.get(k, 0.0) * raw_scores[k] for k in self.METRIC_NAMES)

    # -- public API -----------------------------------------------------------

    def rank_single_mutations(self, vhh_sequence: VHHSequence, off_limits: set = None,
                              forbidden_substitutions: Optional[dict] = None,
                              excluded_target_aas: Optional[set] = None) -> pd.DataFrame:
        if off_limits is None:
            off_limits = set()
        if forbidden_substitutions is None:
            forbidden_substitutions = {}
        suggestions = self.humanness_scorer.get_mutation_suggestions(
            vhh_sequence, off_limits, forbidden_substitutions=forbidden_substitutions,
            excluded_target_aas=excluded_target_aas,
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
                         min_mutations: int = 1,
                         strategy: str = "auto",
                         anchor_threshold: float = 0.6,
                         max_rounds: int = 5) -> pd.DataFrame:
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
        strategy:
            Sampling strategy to use.  One of:

            * ``"auto"`` (default) – exhaustive for small spaces, random for
              medium spaces, iterative refinement for very large spaces
              (> ``_ITERATIVE_THRESHOLD`` combinations).
            * ``"random"`` – always use random sampling.
            * ``"iterative"`` – always use iterative refinement / anchor-and-explore.
        anchor_threshold:
            Fraction of the top-quartile variants a position-mutation pair
            must appear in to be soft-locked as an anchor (default 0.6).
            Only used when ``strategy`` is ``"iterative"`` or ``"auto"`` on
            a very large space.
        max_rounds:
            Maximum number of refinement rounds (default 5).  Only used for
            iterative strategies.
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

        if strategy == "iterative":
            rows = self._generate_iterative(
                vhh_sequence, mut_list, n_mutations, min_mutations,
                max_variants, anchor_threshold=anchor_threshold,
                max_rounds=max_rounds,
            )
        elif strategy == "random":
            rows = self._generate_sampled(
                vhh_sequence, mut_list, n_mutations, min_mutations,
                max_variants,
            )
        else:
            # "auto" strategy
            if total_combos <= _SAMPLING_THRESHOLD:
                rows = self._generate_exhaustive(
                    vhh_sequence, mut_list, n_mutations, min_mutations,
                    max_variants,
                )
            elif total_combos <= _ITERATIVE_THRESHOLD:
                logger.info(
                    "Combinatorial space (~%.2e) exceeds threshold; using random sampling.",
                    total_combos,
                )
                rows = self._generate_sampled(
                    vhh_sequence, mut_list, n_mutations, min_mutations,
                    max_variants,
                )
            else:
                logger.info(
                    "Combinatorial space (~%.2e) exceeds iterative threshold; "
                    "using iterative refinement.",
                    total_combos,
                )
                rows = self._generate_iterative(
                    vhh_sequence, mut_list, n_mutations, min_mutations,
                    max_variants, anchor_threshold=anchor_threshold,
                    max_rounds=max_rounds,
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
            "aggregation_score": round(raw_scores["aggregation_score"], 4),
            "charge_balance_score": round(raw_scores["charge_balance_score"], 4),
            "hydrophobic_core_score": round(raw_scores["hydrophobic_core_score"], 4),
            "disulfide_score": round(raw_scores["disulfide_score"], 4),
            "vhh_hallmark_score": round(raw_scores["vhh_hallmark_score"], 4),
            "ptm_liability_score": round(raw_scores["ptm_liability"], 4),
            "clearance_risk_score": round(raw_scores["clearance_risk"], 4),
            "surface_hydrophobicity_score": round(raw_scores["surface_hydrophobicity"], 4),
            "orthogonal_humanness_score": round(raw_scores["orthogonal_humanness"], 4),
            "orthogonal_stability_score": round(raw_scores["orthogonal_stability"], 4),
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

    def _generate_constrained_sampled(
        self,
        vhh_sequence,
        free_muts: list,
        anchor_muts: list,
        n_free: int,
        min_free: int,
        max_variants: int,
        seen_combos: set,
        variant_counter_start: int,
    ) -> list:
        """Sample variants with *anchor_muts* fixed and *free_muts* randomly varied.

        Parameters
        ----------
        free_muts:
            Candidate mutations for the freely-varied positions.
        anchor_muts:
            Mutation objects that are always included in every variant.
        n_free:
            Maximum number of additional (free) mutations to add.
        min_free:
            Minimum number of free mutations to add.
        max_variants:
            Maximum number of new variants to return.
        seen_combos:
            Mutable set of already-generated combo keys (frozensets).  The
            method adds new keys to this set in-place.
        variant_counter_start:
            Starting index for ``variant_id`` numbering.
        """
        pos_muts: dict[int, list] = {}
        for m in free_muts:
            pos_muts.setdefault(m.imgt_pos, []).append(m)
        unique_free_positions = list(pos_muts.keys())

        rows: list = []
        variant_counter = variant_counter_start
        max_attempts = max_variants * 20

        for _ in range(max_attempts):
            if len(rows) >= max_variants:
                break

            if unique_free_positions:
                k_free = random.randint(
                    max(0, min_free),
                    min(n_free, len(unique_free_positions)),
                )
                chosen_positions = random.sample(unique_free_positions, k_free)
                free_selected = [
                    random.choice(pos_muts[pos])
                    for pos in sorted(chosen_positions)
                ]
            else:
                free_selected = []

            selected = anchor_muts + free_selected
            if not selected:
                continue

            combo_key = frozenset((m.imgt_pos, m.suggested_aa) for m in selected)
            if combo_key in seen_combos:
                continue
            seen_combos.add(combo_key)

            row = self._build_variant_row(vhh_sequence, selected, variant_counter)
            if row is not None:
                rows.append(row)
                variant_counter += 1

        return rows

    def _generate_iterative(
        self,
        vhh_sequence,
        mut_list: list,
        n_mutations: int,
        min_mutations: int,
        max_variants: int,
        anchor_threshold: float = 0.6,
        max_rounds: int = 5,
    ) -> list:
        """Anchor-and-explore iterative refinement strategy.

        Algorithm
        ---------
        1. **Seed round**: random sampling to build an initial variant pool.
        2. **Identify anchors**: position-mutation pairs appearing in
           ``>= anchor_threshold`` fraction of the top-quartile variants
           are soft-locked.
        3. **Constrained re-sampling**: fix anchor mutations, freely vary
           the remaining mutable positions.
        4. **Re-evaluate anchors**: if an anchor position decreases the
           average score in the new combinatorial context it is unlocked.
        5. **Converge**: stop when the top-K pool score stops improving for
           two consecutive rounds or ``max_rounds`` is reached.
        6. **Return**: union of all discovered variants, sorted and capped
           at ``max_variants``.
        """
        # --- Seed round -------------------------------------------------------
        seed_rows = self._generate_sampled(
            vhh_sequence, mut_list, n_mutations, min_mutations, max_variants,
        )
        if not seed_rows:
            return []

        all_rows: list = list(seed_rows)
        seen_combos: set[frozenset] = {
            frozenset(_parse_mut_str(r["mutations"])) for r in all_rows
        }

        K = min(10, len(seed_rows))
        prev_top_k_score = -float("inf")
        stagnant_rounds = 0
        current_mut_list = list(mut_list)

        for _round in range(max_rounds):
            # Sort pool by combined score
            pool_sorted = sorted(
                all_rows, key=lambda r: r["combined_score"], reverse=True
            )
            top_k_avg = sum(r["combined_score"] for r in pool_sorted[:K]) / K

            # Convergence check
            if top_k_avg <= prev_top_k_score + _CONVERGENCE_THRESHOLD:
                stagnant_rounds += 1
                if stagnant_rounds >= 2:
                    logger.info("Iterative refinement converged after %d rounds.", _round)
                    break
            else:
                stagnant_rounds = 0
            prev_top_k_score = top_k_avg

            # Identify top quartile
            q_size = max(1, len(pool_sorted) // 4)
            top_q = pool_sorted[:q_size]

            # Count position-AA frequencies in top quartile (one count per variant)
            pos_aa_counts: dict[tuple[int, str], int] = {}
            for row in top_q:
                for pos, aa in _parse_mut_str(row["mutations"]):
                    pos_aa_counts[(pos, aa)] = pos_aa_counts.get((pos, aa), 0) + 1

            # Soft-lock positions appearing in >= threshold fraction
            # (keep only the highest-frequency AA per position)
            anchors: dict[int, str] = {}  # pos -> aa
            pos_best: dict[int, tuple[int, str]] = {}  # pos -> (count, aa)
            for (pos, aa), count in pos_aa_counts.items():
                freq = count / q_size
                if freq >= anchor_threshold:
                    if pos not in pos_best or count > pos_best[pos][0]:
                        pos_best[pos] = (count, aa)
                        anchors[pos] = aa

            if not anchors:
                logger.info("No anchors found at round %d; stopping early.", _round)
                break

            # Resolve anchor mutation objects
            anchor_muts = [
                m for m in current_mut_list
                if m.imgt_pos in anchors and m.suggested_aa == anchors[m.imgt_pos]
            ]
            if not anchor_muts:
                break

            anchored_positions = {m.imgt_pos for m in anchor_muts}
            free_muts = [m for m in current_mut_list if m.imgt_pos not in anchored_positions]

            if not free_muts:
                break

            n_free = max(1, n_mutations - len(anchor_muts))
            min_free = max(0, min_mutations - len(anchor_muts))

            # Generate constrained variants (anchors fixed, free positions sampled)
            new_rows = self._generate_constrained_sampled(
                vhh_sequence, free_muts, anchor_muts,
                n_free, min_free, max_variants,
                seen_combos, len(all_rows),
            )

            if not new_rows:
                break

            # --- Re-evaluate anchors ------------------------------------------
            # For each anchor position, generate a small batch without it and
            # compare average scores.  If variants without the anchor score
            # higher on average (by > _ANCHOR_UNLOCK_THRESHOLD), unlock that anchor.
            anchors_to_unlock: set[int] = set()
            for anchor_m in anchor_muts:
                apos = anchor_m.imgt_pos
                # Build a free list that includes the anchor position
                unlock_free = free_muts + [anchor_m]
                remaining_anchors = [m for m in anchor_muts if m.imgt_pos != apos]
                sample_size = min(50, max_variants // 5)
                unlock_rows = self._generate_constrained_sampled(
                    vhh_sequence, unlock_free, remaining_anchors,
                    n_free + 1, max(0, min_free), sample_size,
                    set(),  # temporary seen set, don't pollute the global one
                    len(all_rows) + len(new_rows),
                )
                if unlock_rows:
                    avg_constrained = (
                        sum(r["combined_score"] for r in new_rows) / len(new_rows)
                    )
                    avg_unlocked = (
                        sum(r["combined_score"] for r in unlock_rows) / len(unlock_rows)
                    )
                    if avg_unlocked > avg_constrained + _ANCHOR_UNLOCK_THRESHOLD:
                        anchors_to_unlock.add(apos)
                        logger.info(
                            "Unlocking anchor %s%d%s (avg score %.4f → %.4f).",
                            anchor_m.original_aa, apos, anchor_m.suggested_aa,
                            avg_constrained, avg_unlocked,
                        )

            all_rows.extend(new_rows)
            seen_combos.update(
                frozenset(_parse_mut_str(r["mutations"])) for r in new_rows
            )

            # Unlocked anchor positions are re-incorporated implicitly: since the
            # pool now contains variants both with and without those positions,
            # the anchors will not meet the frequency threshold in the next round
            # and will be freely sampled alongside non-anchored positions.
            if anchors_to_unlock:
                logger.info(
                    "Unlocked %d anchor position(s): %s — they will be freely "
                    "varied in subsequent rounds.",
                    len(anchors_to_unlock), sorted(anchors_to_unlock),
                )

        # --- Final deduplication and ranking ----------------------------------
        final_sorted = sorted(all_rows, key=lambda r: r["combined_score"], reverse=True)
        seen_muts: set[str] = set()
        result: list = []
        for r in final_sorted:
            key = r["mutations"]
            if key not in seen_muts:
                seen_muts.add(key)
                result.append(r)

        return result[:max_variants]


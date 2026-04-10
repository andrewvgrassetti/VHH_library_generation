"""Additional developability metrics for VHH antibody sequences.

Provides three independent scoring modules that complement humanness and
stability scoring:

1. **PTM Liability Score** – penalises sequence motifs prone to
   post-translational modifications (deamidation, isomerization,
   oxidation, N-linked glycosylation).
2. **Clearance Risk Score** – estimates in-vivo clearance risk based on
   isoelectric-point deviation from the ideal therapeutic window.
3. **Surface Hydrophobicity Index** – quantifies surface-exposed
   hydrophobic patches that correlate with aggregation and polyreactivity.
"""

import re
from vhh_library.sequence import VHHSequence
from vhh_library.utils import KYTE_DOOLITTLE, sliding_window


# ---------------------------------------------------------------------------
# PTM Liability
# ---------------------------------------------------------------------------

# Motifs and per-occurrence penalties (higher = worse)
_PTM_MOTIFS = [
    # Deamidation-prone: NG, NS, NH (asparagine followed by small/flexible AA)
    (re.compile(r"N[GS]"), "deamidation", 0.15),
    (re.compile(r"NH"), "deamidation", 0.10),
    # Isomerisation-prone: DG, DS, DT, DD
    (re.compile(r"D[GSTD]"), "isomerization", 0.12),
    # Oxidation-prone methionine (not in conserved structural role)
    (re.compile(r"M"), "oxidation", 0.05),
    # N-linked glycosylation motif N-X-[ST] where X != P
    (re.compile(r"N[^P][ST]"), "glycosylation", 0.20),
]

# Maximum raw penalty sum used to normalise to 0-1
_PTM_MAX_RAW = 3.0


class PTMLiabilityScorer:
    """Score a VHH sequence for post-translational modification liabilities."""

    def score(self, vhh_sequence: VHHSequence) -> dict:
        seq = vhh_sequence.sequence
        warnings: list[str] = []
        hits: list[dict] = []
        raw_penalty = 0.0

        for pattern, category, penalty in _PTM_MOTIFS:
            for match in pattern.finditer(seq):
                raw_penalty += penalty
                hits.append({
                    "motif": match.group(),
                    "position": match.start() + 1,  # 1-based
                    "category": category,
                    "penalty": penalty,
                })
                warnings.append(
                    f"{category.capitalize()} motif '{match.group()}' at position {match.start() + 1}"
                )

        # Composite: 1.0 = no liabilities, 0.0 = many liabilities
        composite_score = max(0.0, min(1.0, 1.0 - raw_penalty / _PTM_MAX_RAW))

        return {
            "composite_score": round(composite_score, 4),
            "raw_penalty": round(raw_penalty, 4),
            "hits": hits,
            "warnings": warnings,
        }

    def predict_mutation_effect(self, vhh_sequence: VHHSequence,
                                position: int, new_aa: str) -> float:
        """Return Δ composite score when mutating *position* → *new_aa*."""
        original_score = self.score(vhh_sequence)["composite_score"]
        seq_list = list(vhh_sequence.sequence)
        idx = position - 1
        if idx < 0 or idx >= len(seq_list):
            return 0.0
        old_aa = seq_list[idx]
        if old_aa == new_aa:
            return 0.0
        seq_list[idx] = new_aa
        mutant = VHHSequence("".join(seq_list))
        new_score = self.score(mutant)["composite_score"]
        return round(new_score - original_score, 4)


# ---------------------------------------------------------------------------
# Clearance Risk (pI-based)
# ---------------------------------------------------------------------------

# Ideal pI range for therapeutic antibodies (typical 6–9)
_PI_IDEAL_LOW = 6.0
_PI_IDEAL_HIGH = 9.0
_PI_MAX_DEVIATION = 4.0  # normalisation ceiling


class ClearanceRiskScorer:
    """Estimate clearance risk from isoelectric-point deviation."""

    def __init__(self):
        from vhh_library.stability import StabilityScorer
        self._stability = StabilityScorer()

    def score(self, vhh_sequence: VHHSequence) -> dict:
        stability_result = self._stability.score(vhh_sequence)
        pI = stability_result["pI"]
        net_charge = stability_result["net_charge"]
        warnings: list[str] = []

        if pI < _PI_IDEAL_LOW:
            deviation = _PI_IDEAL_LOW - pI
            warnings.append(f"pI {pI:.1f} is below ideal range ({_PI_IDEAL_LOW}–{_PI_IDEAL_HIGH}).")
        elif pI > _PI_IDEAL_HIGH:
            deviation = pI - _PI_IDEAL_HIGH
            warnings.append(f"pI {pI:.1f} is above ideal range ({_PI_IDEAL_LOW}–{_PI_IDEAL_HIGH}).")
        else:
            deviation = 0.0

        composite_score = max(0.0, min(1.0, 1.0 - deviation / _PI_MAX_DEVIATION))

        return {
            "composite_score": round(composite_score, 4),
            "pI": round(pI, 2),
            "net_charge": round(net_charge, 4),
            "pI_deviation": round(deviation, 4),
            "warnings": warnings,
        }

    def predict_mutation_effect(self, vhh_sequence: VHHSequence,
                                position: int, new_aa: str) -> float:
        original_score = self.score(vhh_sequence)["composite_score"]
        seq_list = list(vhh_sequence.sequence)
        idx = position - 1
        if idx < 0 or idx >= len(seq_list):
            return 0.0
        old_aa = seq_list[idx]
        if old_aa == new_aa:
            return 0.0
        seq_list[idx] = new_aa
        mutant = VHHSequence("".join(seq_list))
        new_score = self.score(mutant)["composite_score"]
        return round(new_score - original_score, 4)


# ---------------------------------------------------------------------------
# Surface Hydrophobicity Index
# ---------------------------------------------------------------------------

_HYDRO_WINDOW = 7  # sliding-window size
_HYDRO_PATCH_THRESHOLD = 1.5  # avg KD above this flags a patch
_HYDRO_MAX_PATCHES = 5  # normalisation ceiling for patch count


class SurfaceHydrophobicityScorer:
    """Score surface hydrophobicity using sliding-window analysis."""

    def __init__(self):
        self.kd_scale = KYTE_DOOLITTLE

    def score(self, vhh_sequence: VHHSequence) -> dict:
        seq = vhh_sequence.sequence
        warnings: list[str] = []

        # Sliding-window hydrophobicity
        window_scores = sliding_window(
            seq, _HYDRO_WINDOW,
            lambda w: sum(self.kd_scale.get(aa, 0) for aa in w) / _HYDRO_WINDOW,
        )

        if not window_scores:
            return {
                "composite_score": 1.0,
                "max_patch_score": 0.0,
                "n_patches": 0,
                "avg_hydrophobicity": 0.0,
                "warnings": [],
            }

        max_patch = max(window_scores)
        n_patches = sum(1 for s in window_scores if s > _HYDRO_PATCH_THRESHOLD)

        if n_patches > 0:
            warnings.append(
                f"{n_patches} hydrophobic patch(es) detected "
                f"(window avg KD > {_HYDRO_PATCH_THRESHOLD})."
            )

        avg_hydro = sum(self.kd_scale.get(aa, 0) for aa in seq) / len(seq)

        # Composite: 1.0 = no hydrophobic patches, 0.0 = many
        patch_penalty = min(1.0, n_patches / _HYDRO_MAX_PATCHES)
        peak_penalty = min(1.0, max(0.0, (max_patch - 0.5) / 3.5))
        composite_score = max(0.0, min(1.0, 1.0 - 0.6 * patch_penalty - 0.4 * peak_penalty))

        return {
            "composite_score": round(composite_score, 4),
            "max_patch_score": round(max_patch, 4),
            "n_patches": n_patches,
            "avg_hydrophobicity": round(avg_hydro, 4),
            "warnings": warnings,
        }

    def predict_mutation_effect(self, vhh_sequence: VHHSequence,
                                position: int, new_aa: str) -> float:
        original_score = self.score(vhh_sequence)["composite_score"]
        seq_list = list(vhh_sequence.sequence)
        idx = position - 1
        if idx < 0 or idx >= len(seq_list):
            return 0.0
        old_aa = seq_list[idx]
        if old_aa == new_aa:
            return 0.0
        seq_list[idx] = new_aa
        mutant = VHHSequence("".join(seq_list))
        new_score = self.score(mutant)["composite_score"]
        return round(new_score - original_score, 4)

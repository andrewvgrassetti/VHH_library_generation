import math
from vhh_library.sequence import VHHSequence
from vhh_library.utils import KYTE_DOOLITTLE, pKa_VALUES, AA_PROPERTIES, sliding_window

VHH_HALLMARKS = {
    37: {"F", "Y"},
    44: {"E"},
    45: {"R"},
    47: {"G"},
}

CANONICAL_DISULFIDE = (23, 104)

# Minimal BLOSUM62 rows for the amino acids expected at VHH hallmark positions.
# Used to compute continuous hallmark sub-scores via similarity-based interpolation.
# Values are the standard BLOSUM62 log-odds scores.
_BLOSUM62_ROWS = {
    "F": {"A": -2, "R": -3, "N": -3, "D": -3, "C": -2, "Q": -3, "E": -3, "G": -3,
          "H": -1, "I":  0, "L":  0, "K": -3, "M":  0, "F":  6, "P": -4, "S": -2,
          "T": -2, "W":  1, "Y":  3, "V": -1},
    "Y": {"A": -2, "R": -2, "N": -2, "D": -3, "C": -2, "Q": -1, "E": -2, "G": -3,
          "H":  2, "I": -1, "L": -1, "K": -2, "M": -1, "F":  3, "P": -3, "S": -2,
          "T": -2, "W":  2, "Y":  7, "V": -1},
    "E": {"A": -1, "R":  0, "N":  0, "D":  2, "C": -3, "Q":  2, "E":  5, "G": -2,
          "H":  0, "I": -3, "L": -3, "K":  1, "M": -2, "F": -3, "P": -1, "S": -1,
          "T": -1, "W": -3, "Y": -2, "V": -2},
    "R": {"A": -1, "R":  5, "N":  0, "D": -2, "C": -3, "Q":  1, "E":  0, "G": -2,
          "H":  0, "I": -3, "L": -2, "K":  2, "M": -1, "F": -3, "P": -2, "S": -1,
          "T": -1, "W": -3, "Y": -2, "V": -3},
    "G": {"A":  0, "R": -2, "N":  0, "D": -1, "C": -3, "Q": -2, "E": -2, "G":  6,
          "H": -2, "I": -4, "L": -4, "K": -2, "M": -3, "F": -3, "P": -2, "S":  0,
          "T": -2, "W": -2, "Y": -3, "V": -3},
}


def _blosum62_similarity(observed: str, expected_aas) -> float:
    """Return a BLOSUM62-based similarity score in [0, 1].

    For each expected amino acid the log-odds score for *observed* is looked up
    and divided by the self-similarity (diagonal) of that expected amino acid.
    The maximum over all expected amino acids is returned, clamped to [0, 1].
    """
    best = 0.0
    for exp in expected_aas:
        row = _BLOSUM62_ROWS.get(exp)
        if row is None:
            continue
        self_sim = row.get(exp, 1)
        if self_sim <= 0:
            self_sim = 1
        raw = row.get(observed, -4)
        sim = max(0.0, raw / self_sim)
        if sim > best:
            best = sim
    return min(1.0, best)


class StabilityScorer:
    def __init__(self):
        self.kd_scale = KYTE_DOOLITTLE
        self.pka = pKa_VALUES

    def score(self, vhh_sequence: VHHSequence) -> dict:
        seq = vhh_sequence.sequence
        numbered = vhh_sequence.imgt_numbered
        warnings = []

        kd_values = [self.kd_scale.get(aa, 0.0) for aa in seq]
        avg_kd = sum(kd_values) / len(kd_values) if kd_values else 0.0
        hydrophobic_core_score = min(1.0, max(0.0, (avg_kd + 2.0) / 4.0))

        net_charge = self._net_charge(seq, pH=7.4)
        charge_balance_score = max(0.0, 1.0 - abs(net_charge) / 10.0)

        pI = self._calculate_pI(seq)

        window_scores = sliding_window(seq, 5, lambda w: sum(self.kd_scale.get(aa, 0) for aa in w) / 5)
        max_patch = max(window_scores) if window_scores else 0.0
        if max_patch > 3.0:
            warnings.append(f"Potential hydrophobic patch detected (avg KD score {max_patch:.2f} in 5-aa window).")
        aggregation_score = min(1.0, max(0.0, 1.0 - (max_patch - 1.0) / 4.0))

        has_c23 = numbered.get(CANONICAL_DISULFIDE[0], "") == "C"
        has_c104 = numbered.get(CANONICAL_DISULFIDE[1], "") == "C"
        if has_c23 and has_c104:
            disulfide_base = 1.0
        elif has_c23 or has_c104:
            disulfide_base = 0.5
            warnings.append("Only one canonical Cys found; disulfide bond may be incomplete.")
        else:
            disulfide_base = 0.0
            warnings.append("Neither canonical Cys23 nor Cys104 found; canonical disulfide absent.")

        # Continuous adjustment: average KD hydrophobicity of ±2 flanking residues
        # around each present canonical Cys (affects disulfide bond accessibility).
        # Scaled to a small ±0.04 adjustment to maintain granularity.
        flank_kd_sum = 0.0
        flank_count = 0
        for cys_imgt, has_cys in ((CANONICAL_DISULFIDE[0], has_c23), (CANONICAL_DISULFIDE[1], has_c104)):
            if has_cys:
                for offset in (-2, -1, 1, 2):
                    aa = numbered.get(cys_imgt + offset, "")
                    if aa:
                        flank_kd_sum += self.kd_scale.get(aa, 0.0)
                        flank_count += 1
        if flank_count > 0:
            avg_flank_kd = flank_kd_sum / flank_count
            # KD values roughly in [-4.5, 4.5]; map to [-0.04, 0.04]
            flank_adj = avg_flank_kd * (0.04 / 4.5)
        else:
            flank_adj = 0.0
        disulfide_score = min(1.0, max(0.0, disulfide_base + flank_adj))

        # Fractional hallmark scoring using BLOSUM62-based similarity.
        # Each position contributes a continuous value in [0, 1] rather than 0 or 1.
        hallmark_score_sum = 0.0
        for imgt_pos, expected_aas in VHH_HALLMARKS.items():
            aa = numbered.get(imgt_pos, "")
            if aa:
                sim = _blosum62_similarity(aa, expected_aas)
                hallmark_score_sum += sim
                if aa not in expected_aas:
                    warnings.append(f"IMGT position {imgt_pos}: expected {expected_aas}, found '{aa}' (VHH hallmark).")
            # Missing position contributes 0 to the sum
        vhh_hallmark_score = hallmark_score_sum / len(VHH_HALLMARKS)

        composite_score = (
            0.2 * hydrophobic_core_score +
            0.15 * charge_balance_score +
            0.2 * aggregation_score +
            0.25 * disulfide_score +
            0.2 * vhh_hallmark_score
        )

        return {
            "hydrophobic_core_score": round(hydrophobic_core_score, 4),
            "charge_balance_score": round(charge_balance_score, 4),
            "net_charge": round(net_charge, 4),
            "pI": round(pI, 2),
            "aggregation_score": round(aggregation_score, 4),
            "disulfide_score": round(disulfide_score, 4),
            "vhh_hallmark_score": round(vhh_hallmark_score, 4),
            "composite_score": round(min(1.0, max(0.0, composite_score)), 4),
            "warnings": warnings,
        }

    def _net_charge(self, seq: str, pH: float = 7.4) -> float:
        charge = 0.0
        charge += 1.0 / (1.0 + 10 ** (pH - self.pka["N_term"]))
        charge -= 1.0 / (1.0 + 10 ** (self.pka["C_term"] - pH))
        aa_pka = {"D": self.pka["D"], "E": self.pka["E"], "H": self.pka["H"],
                  "C": self.pka["C"], "Y": self.pka["Y"], "K": self.pka["K"], "R": self.pka["R"]}
        positive = {"H", "K", "R"}
        negative = {"D", "E", "C", "Y"}
        for aa in seq:
            if aa in positive:
                pka = aa_pka[aa]
                charge += 1.0 / (1.0 + 10 ** (pH - pka))
            elif aa in negative:
                pka = aa_pka[aa]
                charge -= 1.0 / (1.0 + 10 ** (pka - pH))
        return charge

    def _calculate_pI(self, seq: str) -> float:
        lo, hi = 0.0, 14.0
        for _ in range(1000):
            mid = (lo + hi) / 2.0
            charge = self._net_charge(seq, pH=mid)
            if abs(charge) < 0.001:
                break
            if charge > 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def predict_mutation_effect(self, vhh_sequence: VHHSequence, position: int, new_aa: str) -> float:
        original_score = self.score(vhh_sequence)["composite_score"]
        numbered = dict(vhh_sequence.imgt_numbered)
        old_aa = numbered.get(position, None)
        if old_aa is None or old_aa == new_aa:
            return 0.0
        seq_list = list(vhh_sequence.sequence)
        sequence_index = position - 1
        if sequence_index < 0 or sequence_index >= len(seq_list):
            return 0.0
        seq_list[sequence_index] = new_aa
        new_seq_str = "".join(seq_list)
        from vhh_library.sequence import VHHSequence as _VHH
        mutant = _VHH(new_seq_str)
        new_score = self.score(mutant)["composite_score"]
        return round(new_score - original_score, 4)

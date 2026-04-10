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
            disulfide_score = 1.0
        elif has_c23 or has_c104:
            disulfide_score = 0.5
            warnings.append("Only one canonical Cys found; disulfide bond may be incomplete.")
        else:
            disulfide_score = 0.0
            warnings.append("Neither canonical Cys23 nor Cys104 found; canonical disulfide absent.")

        hallmark_hits = 0
        for imgt_pos, expected_aas in VHH_HALLMARKS.items():
            aa = numbered.get(imgt_pos, "")
            if aa in expected_aas:
                hallmark_hits += 1
            else:
                if aa:
                    warnings.append(f"IMGT position {imgt_pos}: expected {expected_aas}, found '{aa}' (VHH hallmark).")
        vhh_hallmark_score = hallmark_hits / len(VHH_HALLMARKS)

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
        idx = position - 1
        if idx < 0 or idx >= len(seq_list):
            return 0.0
        seq_list[idx] = new_aa
        new_seq_str = "".join(seq_list)
        from vhh_library.sequence import VHHSequence as _VHH
        mutant = _VHH(new_seq_str)
        new_score = self.score(mutant)["composite_score"]
        return round(new_score - original_score, 4)

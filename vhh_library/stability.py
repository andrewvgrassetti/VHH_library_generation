import logging
import math
from vhh_library.sequence import VHHSequence
from vhh_library.utils import KYTE_DOOLITTLE, pKa_VALUES, AA_PROPERTIES, sliding_window

logger = logging.getLogger(__name__)

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

# Typical VHH apparent Tm range used for NanoMelt normalisation.
_NANOMELT_TM_MIN = 45.0   # °C — poor thermostability
_NANOMELT_TM_MAX = 85.0   # °C — excellent thermostability


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


def _nanomelt_available() -> bool:
    """Return True if the nanomelt package is importable."""
    try:
        from nanomelt.model.nanomelt import NanoMeltPredPipe  # type: ignore[import]
        return True
    except ImportError:
        return False


def _predict_nanomelt_tm(sequence: str) -> float:
    """Return predicted Tm (°C) for *sequence* via NanoMelt."""
    from nanomelt.model.nanomelt import NanoMeltPredPipe  # type: ignore[import]
    from Bio.SeqRecord import SeqRecord  # type: ignore[import]
    from Bio.Seq import Seq  # type: ignore[import]

    seq_records = [SeqRecord(Seq(sequence), id="query")]
    result_df = NanoMeltPredPipe(seq_records, do_align=False, ncpus=1)
    return float(result_df["NanoMelt Tm (C)"].iloc[0])


def _esm2_pll_available() -> bool:
    """Return True if ESM-2 pseudo-log-likelihood scoring is available."""
    try:
        import torch  # type: ignore[import]
        import esm  # type: ignore[import]
        return True
    except ImportError:
        return False


# Module-level cache for the ESM-2 model to avoid reloading on every call.
_esm2_cache: dict = {}


def compute_esm2_pll(sequences: list[str]) -> list[float]:
    """Compute ESM-2 pseudo-log-likelihood (PLL) for a list of sequences.

    Each sequence is scored by masking each position one at a time and
    summing the log-probability of the true amino acid under ESM-2
    (``esm2_t6_8M_UR50D`` — the smallest ESM-2 checkpoint for CPU use).

    The model is loaded once and cached for subsequent calls.

    Parameters
    ----------
    sequences:
        List of amino-acid strings.

    Returns
    -------
    List of PLL values (higher = more likely / more stable).
    Raises ``ImportError`` if ``torch`` or ``esm`` are not installed.
    """
    import torch  # type: ignore[import]
    import esm  # type: ignore[import]

    if "model" not in _esm2_cache:
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        model.eval()
        _esm2_cache["model"] = model
        _esm2_cache["alphabet"] = alphabet
    model = _esm2_cache["model"]
    alphabet = _esm2_cache["alphabet"]
    batch_converter = alphabet.get_batch_converter()

    pll_scores: list[float] = []

    for seq in sequences:
        data = [("query", seq)]
        _, _, tokens = batch_converter(data)

        seq_len = len(seq)
        log_prob_sum = 0.0

        with torch.no_grad():
            for i in range(1, seq_len + 1):  # skip BOS token at 0
                masked = tokens.clone()
                masked[0, i] = alphabet.mask_idx
                logits = model(masked)["logits"]
                log_probs = torch.nn.functional.log_softmax(logits[0, i], dim=-1)
                true_token = tokens[0, i].item()
                log_prob_sum += log_probs[true_token].item()

        pll_scores.append(round(log_prob_sum, 4))

    return pll_scores


class StabilityScorer:
    """VHH stability scorer.

    By default, uses NanoMelt predicted Tm as the composite stability score
    (normalised to [0, 1]).  When NanoMelt is not installed, falls back to
    the legacy biophysical heuristic composite.

    The sub-scores (hydrophobic core, charge balance, aggregation, disulfide,
    VHH hallmarks) are always computed and included in the result dict for
    diagnostic purposes.
    """

    def __init__(self, *, use_nanomelt: bool = True):
        self.kd_scale = KYTE_DOOLITTLE
        self.pka = pKa_VALUES
        self._use_nanomelt = use_nanomelt and _nanomelt_available()
        if use_nanomelt and not self._use_nanomelt:
            logger.info(
                "NanoMelt not installed; StabilityScorer falling back to "
                "legacy biophysical composite."
            )

    @property
    def nanomelt_active(self) -> bool:
        """True if NanoMelt is being used for the composite score."""
        return self._use_nanomelt

    def score(self, vhh_sequence: VHHSequence) -> dict:
        seq = vhh_sequence.sequence
        numbered = vhh_sequence.imgt_numbered
        warnings = []

        # --- sub-scores (always computed) ---
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
            flank_adj = avg_flank_kd * (0.04 / 4.5)
        else:
            flank_adj = 0.0
        disulfide_score = min(1.0, max(0.0, disulfide_base + flank_adj))

        hallmark_score_sum = 0.0
        for imgt_pos, expected_aas in VHH_HALLMARKS.items():
            aa = numbered.get(imgt_pos, "")
            if aa:
                sim = _blosum62_similarity(aa, expected_aas)
                hallmark_score_sum += sim
                if aa not in expected_aas:
                    warnings.append(f"IMGT position {imgt_pos}: expected {expected_aas}, found '{aa}' (VHH hallmark).")
        vhh_hallmark_score = hallmark_score_sum / len(VHH_HALLMARKS)

        # --- composite score ---
        result: dict = {
            "hydrophobic_core_score": round(hydrophobic_core_score, 4),
            "charge_balance_score": round(charge_balance_score, 4),
            "net_charge": round(net_charge, 4),
            "pI": round(pI, 2),
            "aggregation_score": round(aggregation_score, 4),
            "disulfide_score": round(disulfide_score, 4),
            "vhh_hallmark_score": round(vhh_hallmark_score, 4),
            "warnings": warnings,
        }

        if self._use_nanomelt:
            try:
                tm = _predict_nanomelt_tm(seq)
                normalized = min(
                    1.0,
                    max(0.0, (tm - _NANOMELT_TM_MIN) / (_NANOMELT_TM_MAX - _NANOMELT_TM_MIN)),
                )
                result["composite_score"] = round(normalized, 4)
                result["predicted_tm"] = round(tm, 2)
                result["scoring_method"] = "nanomelt"
            except Exception as exc:
                logger.warning("NanoMelt prediction failed (%s); using legacy composite.", exc)
                legacy = self._legacy_composite(
                    hydrophobic_core_score, charge_balance_score,
                    aggregation_score, disulfide_score, vhh_hallmark_score,
                )
                result["composite_score"] = round(min(1.0, max(0.0, legacy)), 4)
                result["scoring_method"] = "legacy"
        else:
            legacy = self._legacy_composite(
                hydrophobic_core_score, charge_balance_score,
                aggregation_score, disulfide_score, vhh_hallmark_score,
            )
            result["composite_score"] = round(min(1.0, max(0.0, legacy)), 4)
            result["scoring_method"] = "legacy"

        return result

    @staticmethod
    def _legacy_composite(hydrophobic_core: float, charge_balance: float,
                          aggregation: float, disulfide: float,
                          hallmark: float) -> float:
        """Original heuristic composite used as fallback."""
        return (
            0.2 * hydrophobic_core +
            0.15 * charge_balance +
            0.2 * aggregation +
            0.25 * disulfide +
            0.2 * hallmark
        )

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

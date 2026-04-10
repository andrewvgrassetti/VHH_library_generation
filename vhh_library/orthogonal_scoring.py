"""Orthogonal humanness and stability scoring for VHH sequences.

These scorers use methodologies completely independent from the primary
:class:`HumAnnotator` and :class:`StabilityScorer` classes, enabling
cross-validation of the built-in scoring pipeline.

Three scorers are provided:

1. **HumanStringContentScorer** – Humanness metric based on *Human String
   Content* (HSC).  The VHH framework sequence is decomposed into
   overlapping k-mer peptide fragments and each fragment is checked
   against a reference set built from human VH germline frameworks.
   The score equals the fraction of query k-mers that appear in the
   human reference, making it position-independent and orthogonal to
   the position-frequency-matrix approach used by :class:`HumAnnotator`.

2. **ConsensusStabilityScorer** – Stability metric derived from VHH
   germline consensus.  For each framework position the amino acid is
   compared against the most frequent residue observed across known
   camelid VHH germlines.  Positions that match the consensus are
   scored higher, weighted by the conservation level (entropy-based).
   This purely evolutionary / statistical approach is orthogonal to the
   biophysical-property scoring in :class:`StabilityScorer`.

3. **NanoMeltStabilityScorer** – Continuous orthogonal stability scoring
   via NanoMelt predicted melting temperature (Tm).  Uses an ensemble of
   ESM-based and physicochemical embeddings trained on nanobody Tm data.
   Requires the optional ``nanomelt`` package (``pip install nanomelt``).
   The scorer is lazy-loaded; the heavy model weights are only loaded on
   first use.  If ``nanomelt`` is not installed the scorer is unavailable
   and :meth:`NanoMeltStabilityScorer.score` raises :exc:`ImportError`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from vhh_library.sequence import VHHSequence


# ---------------------------------------------------------------------------
# Human String Content (HSC) Scorer
# ---------------------------------------------------------------------------

_DEFAULT_KMER_SIZE = 9


class HumanStringContentScorer:
    """Score humanness via Human String Content (k-mer overlap).

    Parameters
    ----------
    data_dir:
        Directory containing ``human_vh_germlines.json``.
    kmer_size:
        Length of the peptide fragments (default 9).
    """

    def __init__(self, data_dir: Optional[str | Path] = None, kmer_size: int = _DEFAULT_KMER_SIZE):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        data_dir = Path(data_dir)

        with open(data_dir / "human_vh_germlines.json") as fh:
            data = json.load(fh)

        self.kmer_size = kmer_size
        self._reference_kmers = self._build_reference(data["germlines"])

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _extract_framework_seq(germline: dict) -> str:
        """Concatenate framework regions from a germline entry."""
        parts: list[str] = []
        for key in ("fr1", "fr2", "fr3", "fr4"):
            region = germline.get(key, "")
            if region:
                parts.append(region)
        return "".join(parts)

    def _build_reference(self, germlines: list[dict]) -> frozenset[str]:
        """Build the set of all k-mers from human VH germline frameworks."""
        kmers: set[str] = set()
        for germ in germlines:
            fw_seq = self._extract_framework_seq(germ)
            for i in range(len(fw_seq) - self.kmer_size + 1):
                kmers.add(fw_seq[i : i + self.kmer_size])
        return frozenset(kmers)

    # -- public API ----------------------------------------------------------

    def score(self, vhh_sequence: VHHSequence) -> dict:
        """Return Human String Content score for *vhh_sequence*.

        Returns
        -------
        dict with keys:
            ``composite_score`` (float, 0-1): fraction of query k-mers
                found in the human reference.
            ``total_kmers`` (int): number of query k-mers evaluated.
            ``matched_kmers`` (int): number that matched the reference.
        """
        # Build framework string from IMGT-numbered framework positions
        fw_residues = {
            pos: aa
            for pos, aa in vhh_sequence.imgt_numbered.items()
            if pos in vhh_sequence.framework_positions
        }
        fw_seq = "".join(aa for _, aa in sorted(fw_residues.items()))

        if len(fw_seq) < self.kmer_size:
            return {"composite_score": 0.0, "total_kmers": 0, "matched_kmers": 0}

        query_kmers = [fw_seq[i : i + self.kmer_size] for i in range(len(fw_seq) - self.kmer_size + 1)]
        total = len(query_kmers)
        matched = sum(1 for km in query_kmers if km in self._reference_kmers)

        composite = matched / total if total > 0 else 0.0

        return {
            "composite_score": round(min(1.0, max(0.0, composite)), 4),
            "total_kmers": total,
            "matched_kmers": matched,
        }

    def predict_mutation_effect(self, vhh_sequence: VHHSequence, position: int, new_aa: str) -> float:
        """Return Δ composite score when mutating *position* → *new_aa*."""
        original_score = self.score(vhh_sequence)["composite_score"]
        seq_list = list(vhh_sequence.sequence)
        idx = position - 1
        if idx < 0 or idx >= len(seq_list):
            return 0.0
        if seq_list[idx] == new_aa:
            return 0.0
        seq_list[idx] = new_aa
        mutant = VHHSequence("".join(seq_list))
        new_score = self.score(mutant)["composite_score"]
        return round(new_score - original_score, 4)


# ---------------------------------------------------------------------------
# Consensus Stability Scorer
# ---------------------------------------------------------------------------


class ConsensusStabilityScorer:
    """Score stability via VHH germline consensus matching.

    For each framework position the most frequent amino acid across known
    camelid VHH germlines is determined (the *consensus residue*).  The
    query sequence is then scored by the fraction of framework positions
    that match the consensus, weighted by conservation level (positions
    where the consensus is highly conserved count more).

    Parameters
    ----------
    data_dir:
        Directory containing ``vhh_germlines.json``.
    """

    def __init__(self, data_dir: Optional[str | Path] = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        data_dir = Path(data_dir)

        with open(data_dir / "vhh_germlines.json") as fh:
            data = json.load(fh)

        self._consensus, self._conservation = self._build_consensus(data["germlines"])

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _build_consensus(germlines: list[dict]) -> tuple[dict[int, str], dict[int, float]]:
        """Build per-position consensus AA and conservation weight.

        Returns
        -------
        (consensus, conservation):
            *consensus* maps IMGT position → most-frequent amino acid.
            *conservation* maps IMGT position → weight in [0, 1] derived
            from the Shannon entropy of the position.
        """
        from vhh_library.sequence import IMGT_REGIONS

        # Determine which IMGT positions are framework
        framework_ranges = []
        for name, (start, end) in IMGT_REGIONS.items():
            if name.startswith("FR"):
                framework_ranges.append((start, end))

        # Count amino acids at each framework position across germlines
        pos_counts: dict[int, dict[str, int]] = {}
        for germ in germlines:
            # Rebuild position-level mapping from germline FR strings
            for key in ("fr1", "fr2", "fr3", "fr4"):
                region_seq = germ.get(key, "")
                region_name = key.upper()
                if region_name in IMGT_REGIONS:
                    start, _end = IMGT_REGIONS[region_name]
                else:
                    continue
                for i, aa in enumerate(region_seq):
                    imgt_pos = start + i
                    if imgt_pos not in pos_counts:
                        pos_counts[imgt_pos] = {}
                    pos_counts[imgt_pos][aa] = pos_counts[imgt_pos].get(aa, 0) + 1

        consensus: dict[int, str] = {}
        conservation: dict[int, float] = {}

        for pos, counts in pos_counts.items():
            total = sum(counts.values())
            # Consensus = most frequent AA
            best_aa = max(counts, key=lambda a: counts[a])
            consensus[pos] = best_aa

            # Conservation weight derived from Shannon entropy
            # entropy = 0 → perfectly conserved (weight = 1)
            # entropy = log2(20) ≈ 4.32 → fully random (weight ≈ 0)
            entropy = 0.0
            for aa, cnt in counts.items():
                p = cnt / total
                if p > 0:
                    entropy -= p * math.log2(p)
            max_entropy = math.log2(20)  # maximum possible for 20 amino acids
            conservation[pos] = max(0.0, 1.0 - entropy / max_entropy)

        return consensus, conservation

    # -- public API ----------------------------------------------------------

    def score(self, vhh_sequence: VHHSequence) -> dict:
        """Return consensus stability score for *vhh_sequence*.

        Returns
        -------
        dict with keys:
            ``composite_score`` (float, 0-1): weighted fraction of
                framework positions matching the VHH consensus.
            ``positions_evaluated`` (int): number of framework positions
                scored.
            ``consensus_matches`` (int): number of positions matching
                the consensus residue.
            ``avg_conservation`` (float): mean conservation weight across
                evaluated positions.
        """
        fw_residues = {
            pos: aa
            for pos, aa in vhh_sequence.imgt_numbered.items()
            if pos in vhh_sequence.framework_positions
        }

        weighted_sum = 0.0
        weight_total = 0.0
        matches = 0
        evaluated = 0

        for pos, aa in fw_residues.items():
            if pos not in self._consensus:
                continue
            evaluated += 1
            w = self._conservation.get(pos, 0.5)
            weight_total += w
            if aa == self._consensus[pos]:
                weighted_sum += w
                matches += 1

        composite = weighted_sum / weight_total if weight_total > 0 else 0.0
        avg_cons = (
            sum(self._conservation.get(p, 0.5) for p in fw_residues if p in self._consensus) / evaluated
            if evaluated > 0
            else 0.0
        )

        return {
            "composite_score": round(min(1.0, max(0.0, composite)), 4),
            "positions_evaluated": evaluated,
            "consensus_matches": matches,
            "avg_conservation": round(avg_cons, 4),
        }

    def predict_mutation_effect(self, vhh_sequence: VHHSequence, position: int, new_aa: str) -> float:
        """Return Δ composite score when mutating *position* → *new_aa*."""
        original_score = self.score(vhh_sequence)["composite_score"]
        seq_list = list(vhh_sequence.sequence)
        idx = position - 1
        if idx < 0 or idx >= len(seq_list):
            return 0.0
        if seq_list[idx] == new_aa:
            return 0.0
        seq_list[idx] = new_aa
        mutant = VHHSequence("".join(seq_list))
        new_score = self.score(mutant)["composite_score"]
        return round(new_score - original_score, 4)

# ---------------------------------------------------------------------------
# NanoMelt Stability Scorer
# ---------------------------------------------------------------------------

# Typical VHH apparent Tm range used for normalisation.
_NANOMELT_TM_MIN = 45.0   # °C — poor thermostability
_NANOMELT_TM_MAX = 85.0   # °C — excellent thermostability


class NanoMeltStabilityScorer:
    """Continuous orthogonal stability scoring via NanoMelt Tm prediction.

    NanoMelt is a nanobody-specific thermostability predictor that outputs
    a continuous predicted apparent melting temperature (Tm) in °C using an
    ensemble of ESM-1b, ESM-2, one-hot and VHSE embeddings.

    The ``nanomelt`` package (and its heavy dependencies – PyTorch, ESM model
    weights) are loaded **lazily** — only on the first call to :meth:`score`.
    This ensures that users who do not have ``nanomelt`` installed can still
    use the rest of the library without any import errors.

    Parameters
    ----------
    None.  Model weights are downloaded automatically by ``nanomelt`` on
    first use.

    Notes
    -----
    Requires ``pip install nanomelt``.  NanoMelt pulls in PyTorch and ESM
    model weights (~hundreds of MB to several GB).  The first call to
    :meth:`score` will be slow due to model loading; subsequent calls are
    significantly faster.

    The NanoMelt API used here is ``NanoMeltPredPipe(seq_records, do_align,
    ncpus)`` (``nanomelt.model.nanomelt.NanoMeltPredPipe``), which accepts a
    list of ``Bio.SeqRecord`` objects and returns a DataFrame containing the
    column ``"NanoMelt Tm (C)"``.  If the upstream NanoMelt package changes
    its API this method may need adjustment.
    """

    def __init__(self) -> None:
        # Lazy-loaded on first use
        self._nanomelt_pipe = None
        self._available: bool | None = None

    # -- internal helpers ----------------------------------------------------

    def _load(self) -> None:
        """Attempt to import nanomelt and cache the pipeline callable."""
        if self._available is not None:
            return
        try:
            from nanomelt.model.nanomelt import NanoMeltPredPipe  # type: ignore[import]
            self._nanomelt_pipe = NanoMeltPredPipe
            self._available = True
        except ImportError:
            self._available = False

    def _predict_tm(self, sequence: str) -> float:
        """Return predicted Tm (°C) for *sequence* via NanoMelt."""
        from Bio.SeqRecord import SeqRecord  # type: ignore[import]
        from Bio.Seq import Seq  # type: ignore[import]

        seq_records = [SeqRecord(Seq(sequence), id="query")]
        # do_align=False: skip ANARCI alignment (already handled by VHHSequence)
        result_df = self._nanomelt_pipe(seq_records, do_align=False, ncpus=1)
        return float(result_df["NanoMelt Tm (C)"].iloc[0])

    # -- public API ----------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """``True`` if the ``nanomelt`` package is installed and loadable."""
        self._load()
        return bool(self._available)

    def score(self, vhh_sequence: VHHSequence) -> dict:
        """Return NanoMelt stability score for *vhh_sequence*.

        Raises
        ------
        ImportError
            If the ``nanomelt`` package is not installed.

        Returns
        -------
        dict with keys:
            ``composite_score`` (float, 0-1): Tm normalised to
                ``[0, 1]`` using the range
                :data:`_NANOMELT_TM_MIN`-:data:`_NANOMELT_TM_MAX`.
            ``predicted_tm`` (float): raw predicted Tm in °C.
        """
        self._load()
        if not self._available:
            raise ImportError(
                "The 'nanomelt' package is required for NanoMeltStabilityScorer. "
                "Install it with: pip install nanomelt"
            )
        tm = self._predict_tm(vhh_sequence.sequence)
        normalized = min(
            1.0,
            max(0.0, (tm - _NANOMELT_TM_MIN) / (_NANOMELT_TM_MAX - _NANOMELT_TM_MIN)),
        )
        return {
            "composite_score": round(normalized, 4),
            "predicted_tm": round(tm, 2),
        }

    def predict_mutation_effect(self, vhh_sequence: VHHSequence, position: int, new_aa: str) -> float:
        """Return delta composite score when mutating *position* to *new_aa*.

        Returns ``0.0`` if *position* is out of range, if the residue is
        already *new_aa*, or if ``nanomelt`` is not installed.
        """
        if not self.is_available:
            return 0.0
        original = self.score(vhh_sequence)["composite_score"]
        seq_list = list(vhh_sequence.sequence)
        idx = position - 1
        if idx < 0 or idx >= len(seq_list):
            return 0.0
        if seq_list[idx] == new_aa:
            return 0.0
        seq_list[idx] = new_aa
        mutant = VHHSequence("".join(seq_list))
        return round(self.score(mutant)["composite_score"] - original, 4)

"""Trypsin-cleavable peptide barcode generator for LC-MS/MS multiplexed screening."""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vhh_library.utils import tryptic_digest

logger = logging.getLogger(__name__)

# Default path relative to the package data directory
_DEFAULT_POOL_PATH = Path(__file__).parent.parent / "data" / "barcode_pool.json"

# Amino acids allowed in the barcode body (no C = disulfide, no M = oxidation-prone,
# no K/R = internal trypsin cleavage sites that cause incomplete digestion)
_ALLOWED_AAs = list("ADEFGHILNPQSTVWY")
_BASIC_AAs = set("KRH")

# Monoisotopic residue masses (Da)
_MONOISOTOPIC_MASSES: dict[str, float] = {
    "A": 71.03711,  "R": 156.10111, "N": 114.04293, "D": 115.02694,
    "E": 129.04259, "Q": 128.05858, "G": 57.02146,  "H": 137.05891,
    "I": 113.08406, "L": 113.08406, "K": 128.09496, "F": 147.06841,
    "P": 97.05276,  "S": 87.03203,  "T": 101.04768, "W": 186.07931,
    "Y": 163.06333, "V": 99.06841,
}
_WATER = 18.01056
_PROTON = 1.00728

# Fallback mass used for amino acid residues not found in the monoisotopic table
# (approximate average residue mass, Da).
_DEFAULT_AA_MASS = 111.0

# Maximum number of random-generation attempts for a single barcode.
_MAX_GENERATION_ATTEMPTS = 5000


def _barcode_passes_rules(seq: str) -> bool:
    """Return True if *seq* satisfies all barcode design constraints.

    C-terminal barcode orientation: the barcode **starts** with K or R
    (N-terminal tryptic cleavage site) and has no internal K/R in the
    remaining body.
    """
    if not seq:
        return False
    # Length constraint
    if not (6 <= len(seq) <= 12):
        return False
    # Must start with K or R (N-terminal tryptic cleavage site for
    # C-terminal barcodes)
    if seq[0] not in "KR":
        return False
    # No K or R in the body (positions 1 onwards) – internal trypsin
    # cleavage sites would cause incomplete digestion and loss of intact
    # barcode peptide
    if any(aa in "KR" for aa in seq[1:]):
        return False
    # No Met (oxidation-prone) or Cys (disulfide interference)
    if "M" in seq or "C" in seq:
        return False
    # Deamidation-prone dipeptides NG and NS
    if re.search(r"N[GS]", seq):
        return False
    # N-linked glycosylation sequon N-X-S/T where X ≠ P
    for i in range(len(seq) - 2):
        if seq[i] == "N" and seq[i + 1] != "P" and seq[i + 2] in "ST":
            return False
    # At least one basic residue for good MS ionisation (the N-terminal
    # K/R already satisfies this, but check explicitly for clarity)
    if not any(aa in _BASIC_AAs for aa in seq):
        return False
    return True


def _peptide_neutral_mass(seq: str) -> float:
    """Monoisotopic neutral mass of an amino acid sequence (+ water)."""
    return sum(_MONOISOTOPIC_MASSES.get(aa, _DEFAULT_AA_MASS) for aa in seq) + _WATER


def _mz(seq: str, z: int = 2) -> float:
    """Calculate m/z for a given charge state *z*."""
    mass = _peptide_neutral_mass(seq)
    return round((mass + z * _PROTON) / z, 4)


def _hydrophobicity(seq: str) -> float:
    """Kyte-Doolittle average hydrophobicity score."""
    _KD: dict[str, float] = {
        "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "E": -3.5, "Q": -3.5,
        "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,  "F": 2.8,
        "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    }
    return sum(_KD.get(aa, 0.0) for aa in seq) / len(seq) if seq else 0.0


def _generate_barcode_algorithmically(exclude: set[str], rng: random.Random | None = None) -> str | None:
    """Generate a single valid barcode sequence not present in *exclude*.

    Barcodes start with K or R (N-terminal tryptic cleavage site for
    C-terminal barcode orientation) followed by a body of allowed amino
    acids.
    """
    if rng is None:
        rng = random.Random()
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        length = rng.randint(6, 12)
        start = rng.choice(["K", "R"])
        body = [rng.choice(_ALLOWED_AAs) for _ in range(length - 1)]
        seq = start + "".join(body)
        if seq not in exclude and _barcode_passes_rules(seq):
            return seq
    return None


class BarcodeGenerator:
    """Generate and assign trypsin-cleavable peptide barcodes for LC-MS/MS screening.

    Parameters
    ----------
    pool_path:
        Path to a JSON file containing the pre-validated barcode pool.
        Defaults to ``data/barcode_pool.json`` bundled with the package.
    """

    def __init__(self, pool_path: str | Path | None = None):
        path = Path(pool_path) if pool_path else _DEFAULT_POOL_PATH
        if path.exists():
            with open(path) as fh:
                raw = json.load(fh)
            self.pool: list[str] = [entry["sequence"] for entry in raw]
            self._pool_sources: dict[str, str] = {
                entry["sequence"]: entry.get("source", "validated_proteomics")
                for entry in raw
            }
            logger.info("Loaded %d barcodes from %s", len(self.pool), path)
        else:
            logger.warning("Barcode pool not found at %s; starting from an empty pool.", path)
            self.pool = []
            self._pool_sources = {}

    # -- public API -----------------------------------------------------------

    def assign_barcodes(
        self,
        library_df: pd.DataFrame,
        top_n: int = 100,
        linker: str = "GGS",
        c_terminal_tail: str = "",
        check_against_sequences: list[str] | None = None,
    ) -> pd.DataFrame:
        """Assign unique barcodes to the top-N variants in *library_df*.

        The function:
        1. Takes the top *top_n* variants by ``combined_score``.
        2. Performs an in silico tryptic digest of all VHH sequences in the
           library (and any additional sequences supplied) to identify
           collision peptides.
        3. Filters the pre-validated pool to remove collisions.
        4. Assigns one unique barcode per selected variant.
        5. If the pool is exhausted, generates additional barcodes
           algorithmically.

        Barcodes follow C-terminal orientation: each barcode starts with K
        or R (N-terminal tryptic cleavage site).  An optional
        *c_terminal_tail* (2-3 amino acids) is appended after the barcode
        to protect against C-terminal endopeptidase cleavage.

        Parameters
        ----------
        library_df:
            DataFrame from ``MutationEngine.generate_library()``.  Must
            contain at least the columns ``combined_score`` and
            ``aa_sequence``.
        top_n:
            Number of top candidates to barcode (by ``combined_score``).
        linker:
            Flexible linker inserted between the VHH and the barcode
            (default ``"GGS"``).
        c_terminal_tail:
            Short amino acid sequence (typically 2-3 residues) appended
            after the barcode to protect against C-terminal endopeptidase
            cleavage.  Should not contain K or R (to avoid creating
            additional trypsin cleavage sites).  Default is empty string.
        check_against_sequences:
            Optional list of additional amino acid sequences to include in
            the collision check (e.g. host proteome representative peptides).

        Returns
        -------
        DataFrame containing the top-N rows with four additional columns:
        ``barcode_id``, ``barcode_peptide``, ``barcoded_sequence``, and
        ``barcode_tryptic_peptide``.
        """
        if library_df is None or len(library_df) == 0:
            return pd.DataFrame()

        required = {"combined_score", "aa_sequence"}
        missing = required - set(library_df.columns)
        if missing:
            raise ValueError(f"library_df is missing required columns: {missing}")

        # Validate c_terminal_tail – must not contain K or R
        tail = c_terminal_tail.upper().strip() if c_terminal_tail else ""
        if tail and any(aa in "KR" for aa in tail):
            raise ValueError(
                "c_terminal_tail must not contain K or R (would create "
                "additional trypsin cleavage sites)."
            )

        n = min(top_n, len(library_df))
        top_df = (
            library_df.nlargest(n, "combined_score").copy().reset_index(drop=True)
        )

        # -- Collision set: tryptic peptides from library + extra sequences --
        all_seqs: list[str] = list(library_df["aa_sequence"].dropna())
        if check_against_sequences:
            all_seqs.extend(str(s) for s in check_against_sequences if s)

        collision_peptides: set[str] = set()
        for seq in all_seqs:
            collision_peptides.update(tryptic_digest(seq, missed_cleavages=0))
            collision_peptides.update(tryptic_digest(seq, missed_cleavages=1))

        # -- Filter pool ------------------------------------------------------
        available = [bc for bc in self.pool if bc not in collision_peptides]
        # Track which barcodes came from the validated pool
        pool_set = set(available)
        logger.info(
            "Pool size after collision filtering: %d / %d",
            len(available), len(self.pool),
        )

        # -- Algorithmically generate additional barcodes if needed -----------
        if len(available) < n:
            rng = random.Random(0)
            exclude = set(collision_peptides) | set(available)
            while len(available) < n:
                bc = _generate_barcode_algorithmically(exclude, rng)
                if bc is None:
                    logger.warning("Could not generate enough unique barcodes.")
                    break
                available.append(bc)
                exclude.add(bc)

        # -- Assign -----------------------------------------------------------
        barcode_ids = []
        barcode_peptides = []
        barcoded_sequences = []
        barcode_tryptic_peptides = []
        barcode_sources = []

        for i, row in top_df.iterrows():
            if i >= len(available):
                barcode_ids.append("")
                barcode_peptides.append("")
                barcoded_sequences.append(row["aa_sequence"])
                barcode_tryptic_peptides.append("")
                barcode_sources.append("")
                continue

            bc = available[i]
            bc_id = f"BC-{i + 1:03d}"
            vhh_seq: str = row["aa_sequence"]

            # Determine source: validated pool or algorithmically generated
            if bc in pool_set:
                source = self._pool_sources.get(bc, "validated_proteomics")
            else:
                source = "algorithmic"

            # Build barcoded construct: VHH + linker + barcode + tail
            # Barcode starts with K/R (C-terminal orientation); optional
            # protective tail is appended at the very end.
            barcoded = vhh_seq + linker + bc + tail

            # The expected tryptic fragment released by trypsin:
            # trypsin cleaves after the K/R at the start of the barcode.
            # The identifiable tryptic fragment is the barcode body (after
            # the leading K/R) plus the protective tail.
            tryptic_frag = bc[1:] + tail

            barcode_ids.append(bc_id)
            barcode_peptides.append(bc)
            barcoded_sequences.append(barcoded)
            barcode_tryptic_peptides.append(tryptic_frag)
            barcode_sources.append(source)

        top_df["barcode_id"] = barcode_ids
        top_df["barcode_peptide"] = barcode_peptides
        top_df["barcoded_sequence"] = barcoded_sequences
        top_df["barcode_tryptic_peptide"] = barcode_tryptic_peptides
        top_df["barcode_source"] = barcode_sources
        return top_df

    # -- output helpers -------------------------------------------------------

    def generate_barcoded_fasta(self, barcoded_df: pd.DataFrame) -> str:
        """Return a FASTA-format string of barcoded sequences.

        Each entry uses the ``variant_id`` (if present) as the header and the
        ``barcoded_sequence`` as the sequence.  Barcoding information is
        appended to the header line.
        """
        lines: list[str] = []
        for _, row in barcoded_df.iterrows():
            vid = row.get("variant_id", "variant")
            bc_id = row.get("barcode_id", "")
            bc_pep = row.get("barcode_peptide", "")
            header = f">{vid}"
            if bc_id:
                header += f" | {bc_id} | barcode={bc_pep}"
            lines.append(header)
            seq = row.get("barcoded_sequence") or row.get("aa_sequence", "")
            lines.append(str(seq))
        return "\n".join(lines)

    def generate_barcode_reference(self, barcoded_df: pd.DataFrame) -> pd.DataFrame:
        """Generate a reference table for MS method development.

        Returns a DataFrame with columns:
        ``variant_id``, ``barcode_id``, ``barcode_peptide``,
        ``barcode_tryptic_peptide``, ``neutral_mass_da``,
        ``mz_1plus``, ``mz_2plus``, ``mz_3plus``,
        ``hydrophobicity``, ``source``.
        """
        rows = []
        for _, row in barcoded_df.iterrows():
            bc = row.get("barcode_peptide", "")
            tryptic = row.get("barcode_tryptic_peptide", "")
            if not bc:
                continue
            target_peptide = tryptic or bc
            rows.append({
                "variant_id": row.get("variant_id", ""),
                "barcode_id": row.get("barcode_id", ""),
                "barcode_peptide": bc,
                "barcode_tryptic_peptide": tryptic,
                "neutral_mass_da": round(_peptide_neutral_mass(target_peptide), 4),
                "mz_1plus": _mz(target_peptide, z=1),
                "mz_2plus": _mz(target_peptide, z=2),
                "mz_3plus": _mz(target_peptide, z=3),
                "hydrophobicity": round(_hydrophobicity(target_peptide), 4),
                "source": row.get("barcode_source", ""),
            })
        return pd.DataFrame(rows)

    def plot_barcode_distributions(
        self,
        ref_table: pd.DataFrame,
    ) -> matplotlib.figure.Figure:
        """Plot biophysical property distributions for barcode QC.

        Generates a three-panel figure showing:
        1. Hydrophobicity (Kyte-Doolittle) distribution – indicates expected
           RP-HPLC retention-time spread.
        2. m/z at charge state 2+ distribution – the dominant ESI charge state
           for tryptic peptides of this size.
        3. Hydrophobicity vs. m/z scatter – highlights how well barcodes
           separate in 2-D LC-MS space.

        Points are colored by source (validated pool vs. algorithmically
        generated) when the ``source`` column is present.

        Parameters
        ----------
        ref_table:
            DataFrame produced by :meth:`generate_barcode_reference`.

        Returns
        -------
        A :class:`matplotlib.figure.Figure` that can be displayed or saved.
        """
        if ref_table is None or ref_table.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No barcode data to plot",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        hydro = ref_table["hydrophobicity"].to_numpy()
        mz2 = ref_table["mz_2plus"].to_numpy()

        has_source = "source" in ref_table.columns
        source_colors = {
            "validated_proteomics": "#2196F3",
            "algorithmic": "#FF9800",
        }

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

        # --- Panel 1: Hydrophobicity histogram ---
        ax = axes[0]
        if has_source:
            for src, color in source_colors.items():
                mask = ref_table["source"] == src
                if mask.any():
                    ax.hist(hydro[mask], bins=15, alpha=0.7, color=color,
                            label=src.replace("_", " "), edgecolor="white")
            ax.legend(fontsize=8)
        else:
            ax.hist(hydro, bins=15, alpha=0.7, color="#2196F3", edgecolor="white")
        ax.set_xlabel("Hydrophobicity (Kyte-Doolittle avg)")
        ax.set_ylabel("Count")
        ax.set_title("Hydrophobicity Distribution")

        # --- Panel 2: m/z 2+ histogram ---
        ax = axes[1]
        if has_source:
            for src, color in source_colors.items():
                mask = ref_table["source"] == src
                if mask.any():
                    ax.hist(mz2[mask], bins=15, alpha=0.7, color=color,
                            label=src.replace("_", " "), edgecolor="white")
            ax.legend(fontsize=8)
        else:
            ax.hist(mz2, bins=15, alpha=0.7, color="#4CAF50", edgecolor="white")
        ax.set_xlabel("m/z (2+ charge state)")
        ax.set_ylabel("Count")
        ax.set_title("m/z Distribution (2+)")

        # --- Panel 3: scatter hydrophobicity vs m/z ---
        ax = axes[2]
        if has_source:
            for src, color in source_colors.items():
                mask = ref_table["source"] == src
                if mask.any():
                    ax.scatter(hydro[mask], mz2[mask], alpha=0.7, s=30,
                               color=color, label=src.replace("_", " "),
                               edgecolors="white", linewidths=0.3)
            ax.legend(fontsize=8)
        else:
            ax.scatter(hydro, mz2, alpha=0.7, s=30, color="#9C27B0",
                       edgecolors="white", linewidths=0.3)
        ax.set_xlabel("Hydrophobicity (Kyte-Doolittle avg)")
        ax.set_ylabel("m/z (2+ charge state)")
        ax.set_title("Hydrophobicity vs. m/z")

        fig.suptitle("Barcode Peptide Biophysical Distributions", fontsize=13,
                     fontweight="bold")
        return fig

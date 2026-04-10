import json
import re
import math
import numpy as np
from pathlib import Path
from vhh_library.utils import calculate_gc_content


RESTRICTION_SITES = {
    "BsaI":  "GGTCTC",
    "BpiI":  "GAAGAC",
    "EcoRI": "GAATTC",
    "BamHI": "GGATCC",
    "NotI":  "GCGGCCGC",
}


class CodonOptimizer:
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data" / "codon_tables"
        data_dir = Path(data_dir)
        self.codon_tables = {}
        for host_file in data_dir.glob("*.json"):
            host = host_file.stem
            with open(host_file) as f:
                self.codon_tables[host] = json.load(f)

    def optimize(self, aa_sequence: str, host: str, strategy: str = "most_frequent") -> dict:
        if host not in self.codon_tables:
            raise ValueError(f"Unknown host '{host}'. Available: {list(self.codon_tables.keys())}")
        table = self.codon_tables[host]
        warnings = []
        dna_codons = []

        for aa in aa_sequence:
            if aa not in table:
                warnings.append(f"No codon data for amino acid '{aa}'; using NNN.")
                dna_codons.append("NNN")
                continue
            codons = table[aa]
            if not codons:
                dna_codons.append("NNN")
                continue

            if strategy == "most_frequent":
                best_codon = max(codons, key=lambda c: codons[c])
                dna_codons.append(best_codon)

            elif strategy == "harmonized":
                codon_list = list(codons.keys())
                freqs = [codons[c] for c in codon_list]
                total = sum(freqs)
                if total == 0:
                    dna_codons.append(codon_list[0])
                else:
                    probs = [f / total for f in freqs]
                    chosen = np.random.choice(codon_list, p=probs)
                    dna_codons.append(chosen)

            elif strategy == "gc_balanced":
                filtered = {c: f for c, f in codons.items() if f >= 0.05}
                if not filtered:
                    filtered = codons
                target = 0.5
                best_codon = min(
                    filtered.keys(),
                    key=lambda c: abs(calculate_gc_content(c) - target)
                )
                dna_codons.append(best_codon)
            else:
                raise ValueError(f"Unknown strategy '{strategy}'.")

        dna_sequence = "".join(dna_codons)
        gc_content = calculate_gc_content(dna_sequence)

        cai = self._calculate_cai(aa_sequence, dna_codons, table)

        flagged_sites = []
        for enzyme, site in RESTRICTION_SITES.items():
            if site in dna_sequence:
                flagged_sites.append(f"{enzyme} ({site}) found in optimized sequence.")

        for base in "ACGT":
            pattern = base * 6
            if pattern in dna_sequence:
                warnings.append(f"Homopolymer run of >5 '{base}' bases detected.")

        return {
            "dna_sequence": dna_sequence,
            "gc_content": round(gc_content, 4),
            "cai": round(cai, 4),
            "warnings": warnings,
            "flagged_sites": flagged_sites,
        }

    def _calculate_cai(self, aa_sequence: str, dna_codons: list, table: dict) -> float:
        log_sum = 0.0
        count = 0
        for aa, codon in zip(aa_sequence, dna_codons):
            if aa not in table or "N" in codon:
                continue
            codons = table[aa]
            max_freq = max(codons.values()) if codons else 1.0
            codon_freq = codons.get(codon, 0.001)
            if max_freq > 0:
                log_sum += math.log(codon_freq / max_freq)
                count += 1
        if count == 0:
            return 0.0
        return math.exp(log_sum / count)

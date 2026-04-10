import math

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

AA_PROPERTIES = {
    "A": {"hydrophobicity": 1.8, "charge": 0, "polar": False, "aromatic": False},
    "C": {"hydrophobicity": 2.5, "charge": 0, "polar": False, "aromatic": False},
    "D": {"hydrophobicity": -3.5, "charge": -1, "polar": True, "aromatic": False},
    "E": {"hydrophobicity": -3.5, "charge": -1, "polar": True, "aromatic": False},
    "F": {"hydrophobicity": 2.8, "charge": 0, "polar": False, "aromatic": True},
    "G": {"hydrophobicity": -0.4, "charge": 0, "polar": False, "aromatic": False},
    "H": {"hydrophobicity": -3.2, "charge": 0.1, "polar": True, "aromatic": True},
    "I": {"hydrophobicity": 4.5, "charge": 0, "polar": False, "aromatic": False},
    "K": {"hydrophobicity": -3.9, "charge": 1, "polar": True, "aromatic": False},
    "L": {"hydrophobicity": 3.8, "charge": 0, "polar": False, "aromatic": False},
    "M": {"hydrophobicity": 1.9, "charge": 0, "polar": False, "aromatic": False},
    "N": {"hydrophobicity": -3.5, "charge": 0, "polar": True, "aromatic": False},
    "P": {"hydrophobicity": -1.6, "charge": 0, "polar": False, "aromatic": False},
    "Q": {"hydrophobicity": -3.5, "charge": 0, "polar": True, "aromatic": False},
    "R": {"hydrophobicity": -4.5, "charge": 1, "polar": True, "aromatic": False},
    "S": {"hydrophobicity": -0.8, "charge": 0, "polar": True, "aromatic": False},
    "T": {"hydrophobicity": -0.7, "charge": 0, "polar": True, "aromatic": False},
    "V": {"hydrophobicity": 4.2, "charge": 0, "polar": False, "aromatic": False},
    "W": {"hydrophobicity": -0.9, "charge": 0, "polar": False, "aromatic": True},
    "Y": {"hydrophobicity": -1.3, "charge": 0, "polar": True, "aromatic": True},
}

KYTE_DOOLITTLE = {aa: AA_PROPERTIES[aa]["hydrophobicity"] for aa in AA_PROPERTIES}

pKa_VALUES = {
    "D": 3.65,
    "E": 4.25,
    "H": 6.00,
    "C": 8.18,
    "Y": 10.07,
    "K": 10.53,
    "R": 12.48,
    "N_term": 8.00,
    "C_term": 3.10,
}

CODON_TABLE = {
    "TTT": "F", "TTC": "F",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I",
    "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y",
    "TAA": "*", "TAG": "*", "TGA": "*",
    "CAT": "H", "CAC": "H",
    "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C",
    "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G", "a": "t", "t": "a", "g": "c", "c": "g"}


def calculate_gc_content(dna: str) -> float:
    if not dna:
        return 0.0
    dna = dna.upper()
    gc = sum(1 for b in dna if b in "GC")
    return gc / len(dna)


def reverse_complement(dna: str) -> str:
    return "".join(COMPLEMENT.get(b, "N") for b in reversed(dna))


def translate(dna: str) -> str:
    dna = dna.upper()
    protein = []
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i+3]
        aa = CODON_TABLE.get(codon, "X")
        if aa == "*":
            break
        protein.append(aa)
    return "".join(protein)


def sliding_window(sequence, window_size: int, func) -> list:
    results = []
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        results.append(func(window))
    return results

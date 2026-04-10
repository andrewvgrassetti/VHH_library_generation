import json
from pathlib import Path
from vhh_library.utils import CODON_TABLE


class TagManager:
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        data_dir = Path(data_dir)
        with open(data_dir / "tag_sequences.json") as f:
            self._tags = json.load(f)

    def get_available_tags(self) -> dict:
        return self._tags

    def build_construct(self, aa_sequence: str, dna_sequence: str, n_tag: str = None,
                        c_tag: str = None, linker: str = "GSGSGS",
                        custom_n_tag: str = None, custom_c_tag: str = None) -> dict:
        components = []
        aa_parts = []
        dna_parts = []
        schematic_parts = []

        def get_tag_aa(tag_name, custom):
            if custom:
                return custom, ""
            if tag_name and tag_name in self._tags:
                return self._tags[tag_name].get("aa_sequence", ""), self._tags[tag_name].get("dna_sequence", "")
            return "", ""

        n_aa, n_dna = get_tag_aa(n_tag, custom_n_tag)
        if n_aa:
            components.append({"name": n_tag or "Custom N-tag", "type": "n_tag", "aa": n_aa})
            aa_parts.append(n_aa)
            dna_parts.append(n_dna)
            schematic_parts.append(f"[{n_tag or 'Custom N-tag'}]")
            linker_dna = self._encode_linker(linker)
            components.append({"name": "Linker", "type": "linker", "aa": linker})
            aa_parts.append(linker)
            dna_parts.append(linker_dna)
            schematic_parts.append(f"--[Linker]--")

        components.append({"name": "VHH", "type": "vhh", "aa": aa_sequence})
        aa_parts.append(aa_sequence)
        dna_parts.append(dna_sequence)
        schematic_parts.append("[VHH]")

        c_aa, c_dna = get_tag_aa(c_tag, custom_c_tag)
        if c_aa:
            linker_dna = self._encode_linker(linker)
            components.append({"name": "Linker", "type": "linker", "aa": linker})
            aa_parts.append(linker)
            dna_parts.append(linker_dna)
            schematic_parts.append(f"--[Linker]--")
            components.append({"name": c_tag or "Custom C-tag", "type": "c_tag", "aa": c_aa})
            aa_parts.append(c_aa)
            dna_parts.append(c_dna)
            schematic_parts.append(f"[{c_tag or 'Custom C-tag'}]")

        return {
            "aa_construct": "".join(aa_parts),
            "dna_construct": "".join(dna_parts),
            "schematic": "".join(schematic_parts),
            "components": components,
        }

    def _encode_linker(self, linker_aa: str) -> str:
        simple_map = {
            "G": "GGT", "S": "TCT", "A": "GCT", "P": "CCT",
            "E": "GAA", "K": "AAA",
        }
        return "".join(simple_map.get(aa, "NNN") for aa in linker_aa)

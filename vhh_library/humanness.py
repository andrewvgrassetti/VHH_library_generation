import json
from pathlib import Path
from vhh_library.sequence import VHHSequence
from vhh_library.utils import AMINO_ACIDS

class HumAnnotator:
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        data_dir = Path(data_dir)
        with open(data_dir / "human_vh_germlines.json") as f:
            data = json.load(f)
        self.germlines = data["germlines"]
        self.pfm = {int(k): v for k, v in data["position_frequency_matrix"].items()}

    def _framework_sequence(self, vhh_sequence: VHHSequence) -> dict:
        return {pos: aa for pos, aa in vhh_sequence.imgt_numbered.items()
                if pos in vhh_sequence.framework_positions}

    def score(self, vhh_sequence: VHHSequence) -> dict:
        fw_residues = self._framework_sequence(vhh_sequence)

        position_scores = {}
        for pos, aa in fw_residues.items():
            if pos in self.pfm:
                freq_dict = self.pfm[pos]
                score = freq_dict.get(aa, freq_dict.get("other", 0.0))
            else:
                score = 0.0
            position_scores[pos] = float(score)

        if position_scores:
            composite_score = sum(position_scores.values()) / len(position_scores)
        else:
            composite_score = 0.0

        best_germline = None
        best_identity = 0.0
        for germ in self.germlines:
            germ_seq = "".join([germ.get("fr1", ""), germ.get("fr2", ""), germ.get("fr3", ""), germ.get("fr4", "")])
            vhh_fw_seq = "".join(fw_residues.get(p, "-") for p in sorted(fw_residues.keys()))
            min_len = min(len(germ_seq), len(vhh_fw_seq))
            if min_len == 0:
                continue
            matches = sum(1 for a, b in zip(germ_seq[:min_len], vhh_fw_seq[:min_len]) if a == b)
            identity = matches / min_len
            if identity > best_identity:
                best_identity = identity
                best_germline = germ["name"]

        position_annotations = {}
        for pos, sc in position_scores.items():
            if sc >= 0.3:
                position_annotations[pos] = "human"
            elif sc >= 0.05:
                position_annotations[pos] = "acceptable"
            else:
                position_annotations[pos] = "non-human"

        return {
            "germline_identity": float(best_identity),
            "best_germline": best_germline or "",
            "position_scores": position_scores,
            "composite_score": min(1.0, max(0.0, float(composite_score))),
            "position_annotations": position_annotations,
        }

    def get_mutation_suggestions(self, vhh_sequence: VHHSequence, off_limits: set,
                                forbidden_substitutions: dict | None = None) -> list:
        """Get mutation suggestions for a VHH sequence.

        Args:
            vhh_sequence: The VHH sequence to analyze.
            off_limits: Set of IMGT positions where no mutations are allowed.
            forbidden_substitutions: Optional dict mapping IMGT position (int) to
                a set of one-letter amino acid codes that are forbidden as targets
                at that position. Mutations to these amino acids will be excluded.
        """
        if forbidden_substitutions is None:
            forbidden_substitutions = {}
        fw_residues = self._framework_sequence(vhh_sequence)
        suggestions = []
        for pos, aa in fw_residues.items():
            if pos in off_limits:
                continue
            if pos not in self.pfm:
                continue
            freq_dict = self.pfm[pos]
            current_score = freq_dict.get(aa, freq_dict.get("other", 0.0))
            pos_forbidden = forbidden_substitutions.get(pos, set())
            for candidate_aa in AMINO_ACIDS:
                if candidate_aa == aa:
                    continue
                if candidate_aa in pos_forbidden:
                    continue
                cand_score = freq_dict.get(candidate_aa, freq_dict.get("other", 0.0))
                delta = cand_score - current_score
                if delta > 0.05:
                    suggestions.append({
                        "position": pos,
                        "original_aa": aa,
                        "suggested_aa": candidate_aa,
                        "delta_humanness": float(delta),
                        "reason": f"Increases humanness at IMGT {pos} from {current_score:.2f} to {cand_score:.2f}",
                    })
        suggestions.sort(key=lambda x: x["delta_humanness"], reverse=True)
        return suggestions

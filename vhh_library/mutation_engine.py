import itertools
import pandas as pd
from vhh_library.humanness import HumAnnotator
from vhh_library.stability import StabilityScorer
from vhh_library.sequence import VHHSequence


class MutationEngine:
    def __init__(self, humanness_scorer: HumAnnotator, stability_scorer: StabilityScorer,
                 w_humanness: float = 0.6, w_stability: float = 0.4):
        self.humanness_scorer = humanness_scorer
        self.stability_scorer = stability_scorer
        self.w_humanness = w_humanness
        self.w_stability = w_stability

    def rank_single_mutations(self, vhh_sequence: VHHSequence, off_limits: set = None,
                              forbidden_substitutions: dict | None = None) -> pd.DataFrame:
        if off_limits is None:
            off_limits = set()
        if forbidden_substitutions is None:
            forbidden_substitutions = {}
        suggestions = self.humanness_scorer.get_mutation_suggestions(
            vhh_sequence, off_limits, forbidden_substitutions=forbidden_substitutions
        )
        rows = []
        for s in suggestions:
            imgt_pos = s["position"]
            delta_h = s["delta_humanness"]
            delta_s = self.stability_scorer.predict_mutation_effect(vhh_sequence, imgt_pos, s["suggested_aa"])
            combined = self.w_humanness * delta_h + self.w_stability * delta_s
            rows.append({
                "position": imgt_pos,
                "imgt_pos": imgt_pos,
                "original_aa": s["original_aa"],
                "suggested_aa": s["suggested_aa"],
                "delta_humanness": round(delta_h, 4),
                "delta_stability": round(delta_s, 4),
                "combined_score": round(combined, 4),
                "reason": s["reason"],
            })
        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
        return df

    def apply_mutations(self, sequence: str, mutations: list) -> str:
        seq_list = list(sequence)
        for idx, new_aa in mutations:
            if 0 <= idx < len(seq_list):
                seq_list[idx] = new_aa
        return "".join(seq_list)

    def generate_library(self, vhh_sequence: VHHSequence, top_mutations: pd.DataFrame,
                         n_mutations: int, max_variants: int = 10000) -> pd.DataFrame:
        if len(top_mutations) == 0:
            return pd.DataFrame()

        candidates = top_mutations.head(min(len(top_mutations), n_mutations * 3))
        mut_list = list(candidates.itertuples(index=False))

        rows = []
        variant_counter = 0

        base_h_score = self.humanness_scorer.score(vhh_sequence)["composite_score"]
        base_s_score = self.stability_scorer.score(vhh_sequence)["composite_score"]

        for k in range(1, n_mutations + 1):
            for combo in itertools.combinations(range(len(mut_list)), k):
                if variant_counter >= max_variants:
                    break
                selected = [mut_list[i] for i in combo]
                positions = [m.imgt_pos for m in selected]
                if len(set(positions)) != len(positions):
                    continue
                muts_0idx = [(m.imgt_pos - 1, m.suggested_aa) for m in selected]
                new_seq = self.apply_mutations(vhh_sequence.sequence, muts_0idx)
                mut_str = ", ".join(f"{m.original_aa}{m.imgt_pos}{m.suggested_aa}" for m in selected)
                delta_h = sum(m.delta_humanness for m in selected)
                delta_s = sum(m.delta_stability for m in selected)
                h_score = min(1.0, max(0.0, base_h_score + delta_h))
                s_score = min(1.0, max(0.0, base_s_score + delta_s))
                combined = self.w_humanness * h_score + self.w_stability * s_score
                rows.append({
                    "variant_id": f"VAR-{variant_counter+1:06d}",
                    "mutations": mut_str,
                    "humanness_score": round(h_score, 4),
                    "stability_score": round(s_score, 4),
                    "combined_score": round(combined, 4),
                    "aa_sequence": new_seq,
                })
                variant_counter += 1
            if variant_counter >= max_variants:
                break

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
        return df

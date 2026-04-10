from vhh_library.sequence import VHHSequence

REGION_COLORS = {
    "FR1":  "#E3F2FD",
    "CDR1": "#FFCDD2",
    "FR2":  "#E8F5E9",
    "CDR2": "#FFCDD2",
    "FR3":  "#E3F2FD",
    "CDR3": "#FFCDD2",
    "FR4":  "#E8F5E9",
}

MUT_COLORS = {
    "humanness": "#4CAF50",
    "stability":  "#2196F3",
    "both":       "#9C27B0",
}


class SequenceVisualizer:
    def render_alignment(self, original: VHHSequence, mutant_aa: str, mutation_info: dict) -> str:
        orig_seq = original.sequence
        cdr_pos = original.cdr_positions

        html_orig = []
        html_mut  = []

        for i, (orig_aa, mut_aa) in enumerate(zip(orig_seq, mutant_aa)):
            imgt_pos = i + 1
            bg_orig = "#FFCDD2" if imgt_pos in cdr_pos else "#F5F5F5"
            if i in mutation_info:
                mut_type = mutation_info[i]
                color = MUT_COLORS.get(mut_type, "#9E9E9E")
            else:
                color = "#9E9E9E"

            html_orig.append(
                f'<span style="background-color:{bg_orig};padding:1px 2px;font-family:monospace">{orig_aa}</span>'
            )
            html_mut.append(
                f'<span style="background-color:{color if i in mutation_info else "#F5F5F5"};'
                f'color:{"white" if i in mutation_info else "#333"};padding:1px 2px;font-family:monospace">{mut_aa}</span>'
            )

        orig_line = "".join(html_orig)
        mut_line  = "".join(html_mut)

        html = (
            '<div style="font-family:monospace;line-height:2">'
            f'<div><b>Original:</b> {orig_line}</div>'
            f'<div><b>Mutant&nbsp;:</b> {mut_line}</div>'
            "</div>"
        )
        return html

    def render_region_track(self, vhh_sequence: VHHSequence) -> str:
        regions = vhh_sequence.regions
        parts = []
        for region_name, (start, end, seq_str) in regions.items():
            if not seq_str:
                continue
            color = REGION_COLORS.get(region_name, "#EEE")
            parts.append(
                f'<span style="background-color:{color};padding:2px 6px;margin:1px;'
                f'border-radius:3px;font-size:0.8em;font-family:monospace">'
                f'<b>{region_name}</b> ({len(seq_str)}aa)</span>'
            )
        return '<div style="display:flex;flex-wrap:wrap;gap:2px">' + "".join(parts) + "</div>"

    def render_score_bar(self, score: float, label: str, color: str = "#4CAF50") -> str:
        pct = int(score * 100)
        html = (
            f'<div style="margin-bottom:6px">'
            f'<span style="font-size:0.9em">{label}</span>'
            f'<div style="background:#eee;border-radius:4px;height:18px;width:100%;margin-top:2px">'
            f'<div style="background:{color};width:{pct}%;height:18px;border-radius:4px;'
            f'display:flex;align-items:center;padding-left:6px;color:white;font-size:0.8em">'
            f'{pct}%</div></div></div>'
        )
        return html

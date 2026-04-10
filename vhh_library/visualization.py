from vhh_library.sequence import VHHSequence, IMGT_REGIONS

REGION_COLORS = {
    "FR1":  "#E3F2FD",
    "CDR1": "#FFCDD2",
    "FR2":  "#E8F5E9",
    "CDR2": "#FFCDD2",
    "FR3":  "#E3F2FD",
    "CDR3": "#FFCDD2",
    "FR4":  "#E8F5E9",
}

REGION_BORDER_COLORS = {
    "FR1":  "#90CAF9",
    "CDR1": "#EF9A9A",
    "FR2":  "#A5D6A7",
    "CDR2": "#EF9A9A",
    "FR3":  "#90CAF9",
    "CDR3": "#EF9A9A",
    "FR4":  "#A5D6A7",
}

REGION_LABEL_COLORS = {
    "FR1":  "#1565C0",
    "CDR1": "#C62828",
    "FR2":  "#2E7D32",
    "CDR2": "#C62828",
    "FR3":  "#1565C0",
    "CDR3": "#C62828",
    "FR4":  "#2E7D32",
}

MUT_COLORS = {
    "humanness": "#4CAF50",
    "stability":  "#2196F3",
    "both":       "#9C27B0",
}

# Notable residue categories for the legend
# IMGT positions for conserved structural features of VHH domains:
#   Cys23 and Cys104 form the canonical intradomain disulfide bond
#   Trp47 (FR2 hallmark) and Trp118 (conserved in FR4) are VHH hallmark residues
NOTABLE_RESIDUES = {
    "Cys (disulfide)": {"positions": {23, 104}, "color": "#FFD600", "text_color": "#000"},
    "Trp (VHH hallmark)": {"positions": {47, 118}, "color": "#AB47BC", "text_color": "#FFF"},
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

    def render_off_limits_track(self, vhh_sequence: VHHSequence, off_limit_positions: set,
                                forbidden_substitutions: dict | None = None) -> str:
        """Render an interactive single-line linear depiction of the VHH.

        Shows each residue as a colored cell with CDR/FR regions labeled above,
        notable residues highlighted, and off-limit positions marked with a
        distinct overlay pattern. Includes a color legend.

        Args:
            vhh_sequence: The VHH sequence to visualize.
            off_limit_positions: Set of IMGT positions that are fully off-limits.
            forbidden_substitutions: Dict mapping IMGT position to set of
                forbidden target amino acids (shown with partial restriction marker).
        """
        if forbidden_substitutions is None:
            forbidden_substitutions = {}

        seq = vhh_sequence.sequence
        numbered = vhh_sequence.imgt_numbered
        seq_len = min(len(seq), 128)

        # Build position-to-region mapping
        pos_to_region = {}
        for region_name, (start, end) in IMGT_REGIONS.items():
            for p in range(start, end + 1):
                pos_to_region[p] = region_name

        # Build notable position set
        notable_positions = {}
        for label, info in NOTABLE_RESIDUES.items():
            for p in info["positions"]:
                if p in numbered:
                    notable_positions[p] = (label, info["color"], info["text_color"])

        # Cell dimensions
        cell_w = 7
        cell_h = 22
        label_h = 28
        legend_h = 32
        total_w = seq_len * cell_w + 20
        total_h = label_h + cell_h + legend_h + 10

        svg_parts = []
        svg_parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" '
            f'style="font-family:Arial,sans-serif;font-size:8px">'
        )

        # --- Region labels above the track ---
        for region_name, (start, end) in IMGT_REGIONS.items():
            actual_start = max(1, start)
            actual_end = min(seq_len, end)
            if actual_start > seq_len:
                continue
            x1 = (actual_start - 1) * cell_w + 10
            x2 = actual_end * cell_w + 10
            mid_x = (x1 + x2) / 2
            color = REGION_LABEL_COLORS.get(region_name, "#333")
            # Region bracket line
            svg_parts.append(
                f'<line x1="{x1}" y1="{label_h - 4}" x2="{x2}" y2="{label_h - 4}" '
                f'stroke="{color}" stroke-width="2"/>'
            )
            # End ticks
            svg_parts.append(
                f'<line x1="{x1}" y1="{label_h - 8}" x2="{x1}" y2="{label_h - 2}" '
                f'stroke="{color}" stroke-width="1.5"/>'
            )
            svg_parts.append(
                f'<line x1="{x2}" y1="{label_h - 8}" x2="{x2}" y2="{label_h - 2}" '
                f'stroke="{color}" stroke-width="1.5"/>'
            )
            # Label text
            svg_parts.append(
                f'<text x="{mid_x}" y="{label_h - 12}" text-anchor="middle" '
                f'fill="{color}" font-size="9px" font-weight="bold">{region_name}</text>'
            )

        # --- Residue cells ---
        y = label_h
        for i in range(seq_len):
            imgt_pos = i + 1
            aa = numbered.get(imgt_pos, "?")
            region = pos_to_region.get(imgt_pos, "")
            x = i * cell_w + 10

            # Background color based on region
            bg_color = REGION_COLORS.get(region, "#F5F5F5")
            border_color = REGION_BORDER_COLORS.get(region, "#CCC")

            # Check notable residue override
            if imgt_pos in notable_positions:
                _, bg_color, _ = notable_positions[imgt_pos]
                border_color = bg_color

            svg_parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                f'fill="{bg_color}" stroke="{border_color}" stroke-width="0.5"/>'
            )

            # Off-limit overlay: diagonal hatch pattern
            if imgt_pos in off_limit_positions:
                svg_parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                    f'fill="rgba(0,0,0,0.35)" stroke="none"/>'
                )
                svg_parts.append(
                    f'<line x1="{x}" y1="{y + cell_h}" x2="{x + cell_w}" y2="{y}" '
                    f'stroke="white" stroke-width="0.8" opacity="0.7"/>'
                )
            elif imgt_pos in forbidden_substitutions and forbidden_substitutions[imgt_pos]:
                # Partial restriction: dot marker
                svg_parts.append(
                    f'<circle cx="{x + cell_w / 2}" cy="{y + cell_h - 3}" r="1.5" '
                    f'fill="#FF6F00"/>'
                )

            # Amino acid letter (only every 10th to avoid clutter at small scale)
            if imgt_pos % 10 == 0 or imgt_pos == 1:
                svg_parts.append(
                    f'<text x="{x + cell_w / 2}" y="{y + cell_h + 9}" '
                    f'text-anchor="middle" fill="#666" font-size="7px">{imgt_pos}</text>'
                )

        # --- Legend ---
        legend_y = y + cell_h + 18
        legend_items = [
            ("FR region", REGION_COLORS["FR1"], REGION_BORDER_COLORS["FR1"], None),
            ("CDR region", REGION_COLORS["CDR1"], REGION_BORDER_COLORS["CDR1"], None),
        ]
        for label, info in NOTABLE_RESIDUES.items():
            legend_items.append((label, info["color"], info["color"], None))
        legend_items.append(("Off-limit", "#999", "#666", "hatch"))
        if forbidden_substitutions:
            legend_items.append(("Partial restriction", "#FFF3E0", "#FF6F00", "dot"))

        lx = 10
        for label_text, fill, stroke, marker_type in legend_items:
            # Color swatch
            svg_parts.append(
                f'<rect x="{lx}" y="{legend_y}" width="12" height="12" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1" rx="2"/>'
            )
            if marker_type == "hatch":
                svg_parts.append(
                    f'<rect x="{lx}" y="{legend_y}" width="12" height="12" '
                    f'fill="rgba(0,0,0,0.35)" rx="2"/>'
                )
                svg_parts.append(
                    f'<line x1="{lx}" y1="{legend_y + 12}" x2="{lx + 12}" y2="{legend_y}" '
                    f'stroke="white" stroke-width="1"/>'
                )
            elif marker_type == "dot":
                svg_parts.append(
                    f'<circle cx="{lx + 6}" cy="{legend_y + 9}" r="2" fill="#FF6F00"/>'
                )
            svg_parts.append(
                f'<text x="{lx + 16}" y="{legend_y + 10}" fill="#333" font-size="9px">'
                f'{label_text}</text>'
            )
            lx += len(label_text) * 6 + 30

        svg_parts.append("</svg>")
        return "".join(svg_parts)

import streamlit as st
import pandas as pd
import io
from pathlib import Path

from vhh_library.sequence import VHHSequence, IMGT_REGIONS
from vhh_library.humanness import HumAnnotator
from vhh_library.stability import StabilityScorer
from vhh_library.mutation_engine import MutationEngine
from vhh_library.codon_optimizer import CodonOptimizer
from vhh_library.tags import TagManager
from vhh_library.library_manager import LibraryManager
from vhh_library.visualization import SequenceVisualizer
from vhh_library.components import sequence_selector

st.set_page_config(
    page_title="VHH Biosimilar Library Generator",
    layout="wide",
    page_icon="🧬",
)

SAMPLE_VHH = "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"


@st.cache_resource
def load_scorers():
    h = HumAnnotator()
    s = StabilityScorer()
    return h, s


def init_state():
    defaults = {
        "vhh_seq": None,
        "humanness_scores": None,
        "stability_scores": None,
        "ranked_mutations": None,
        "library": None,
        "library_manager": LibraryManager(),
        "construct": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def sidebar():
    st.sidebar.header("⚙️ Configuration")

    st.sidebar.subheader("Session Management")
    st.sidebar.write(f"**Session ID:** `{st.session_state.library_manager.session_id}`")
    if st.sidebar.button("New Session"):
        st.session_state.library_manager = LibraryManager()
        st.rerun()

    uploaded = st.sidebar.file_uploader("Load Session (JSON)", type=["json"])
    if uploaded:
        import json
        data = json.load(uploaded)
        st.session_state["loaded_session"] = data
        st.sidebar.success("Session loaded!")

    st.sidebar.subheader("Scoring Weights")
    hw = st.sidebar.slider("Humanness Weight", 0.0, 1.0, 0.6, step=0.05, key="humanness_weight")
    st.sidebar.write(f"Stability Weight: **{1 - hw:.2f}**")

    st.sidebar.subheader("Library Parameters")
    n_mut = st.sidebar.slider("Max Mutations per Variant", 1, 20, 5, key="n_mutations")
    max_var = st.sidebar.number_input("Max Variants", 100, 10000, 1000, step=100, key="max_variants")

    st.sidebar.subheader("Expression System")
    host = st.sidebar.selectbox("Host", ["e_coli", "s_cerevisiae", "p_pastoris"], key="host")
    strategy = st.sidebar.selectbox("Codon Strategy", ["most_frequent", "harmonized", "gc_balanced"], key="strategy")

    return hw, n_mut, max_var, host, strategy


def tab_input(humanness_scorer, stability_scorer, viz):
    st.header("🔬 Input & Analysis")
    seq_input = st.text_area("Paste VHH amino acid sequence:", value=SAMPLE_VHH, height=100)

    if st.button("Analyze Sequence", type="primary"):
        try:
            vhh = VHHSequence(seq_input.strip())
            st.session_state.vhh_seq = vhh
            st.session_state.humanness_scores = humanness_scorer.score(vhh)
            st.session_state.stability_scores = stability_scorer.score(vhh)
        except Exception as e:
            st.error(f"Error analyzing sequence: {e}")

    if st.session_state.vhh_seq is None:
        return

    vhh = st.session_state.vhh_seq
    hs = st.session_state.humanness_scores
    ss = st.session_state.stability_scores

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Length", vhh.length)
    with col2:
        valid = vhh.validation_result["valid"]
        st.metric("Valid", "✅ Yes" if valid else "❌ No")
    with col3:
        st.metric("Warnings", len(vhh.validation_result["warnings"]))

    if vhh.validation_result["errors"]:
        for err in vhh.validation_result["errors"]:
            st.error(err)
    if vhh.validation_result["warnings"]:
        for w in vhh.validation_result["warnings"]:
            st.warning(w)

    st.subheader("Region Map")
    st.components.v1.html(viz.render_region_track(vhh), height=60)

    st.subheader("IMGT Regions")
    region_rows = []
    for rname, (start, end, seq_str) in vhh.regions.items():
        region_rows.append({"Region": rname, "IMGT Start": start, "IMGT End": end,
                             "Length": len(seq_str), "Sequence": seq_str})
    st.dataframe(pd.DataFrame(region_rows), use_container_width=True)

    st.subheader("Humanness Scores")
    st.components.v1.html(
        viz.render_score_bar(hs["composite_score"], "Composite Humanness", "#4CAF50") +
        viz.render_score_bar(hs["germline_identity"], f"Best Germline Identity ({hs['best_germline']})", "#2196F3"),
        height=100,
    )

    st.subheader("Stability Scores")
    score_html = ""
    for key, color in [("composite_score", "#FF9800"), ("disulfide_score", "#9C27B0"),
                       ("vhh_hallmark_score", "#00BCD4"), ("aggregation_score", "#8BC34A")]:
        score_html += viz.render_score_bar(ss[key], key.replace("_", " ").title(), color)
    st.components.v1.html(score_html, height=200)

    col1, col2, col3 = st.columns(3)
    col1.metric("Net Charge (pH 7.4)", f"{ss['net_charge']:.2f}")
    col2.metric("pI", f"{ss['pI']:.2f}")
    if ss["warnings"]:
        with st.expander("Stability Warnings"):
            for w in ss["warnings"]:
                st.warning(w)


def _parse_off_limit_csv(uploaded_file) -> dict:
    """Parse a CSV of per-amino-acid forbidden substitutions.

    Expected CSV format (with or without header):
        Column 1: Single-letter amino acid code (the original residue, e.g. "A")
        Column 2: One-letter amino acid codes that are forbidden as replacements
                   for the amino acid in column 1 (e.g. "VIL" or "V,I,L")

    For example, if column 1 is ``A`` and column 2 is ``VIL``, then any
    position in the VHH sequence that contains ``A`` cannot be mutated to
    ``V``, ``I``, or ``L``.

    Returns:
        dict mapping original amino acid (str, single letter) -> set of
        forbidden replacement one-letter AA codes.
    """
    content = uploaded_file.read().decode("utf-8")
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(io.StringIO(content), header=None)
    except Exception:
        return {}

    if len(df.columns) < 2:
        return {}

    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    forbidden: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        orig_aa = str(row.iloc[0]).strip().upper()
        forbidden_str = str(row.iloc[1]).strip()

        # Column 1 must be a single valid amino acid letter
        if len(orig_aa) != 1 or orig_aa not in valid_aas:
            continue

        # Parse forbidden AAs: could be "VIL" or "V,I,L" or "V I L"
        forbidden_str = forbidden_str.replace(",", "").replace(" ", "").upper()
        forbidden_aas = set(c for c in forbidden_str if c in valid_aas)
        if forbidden_aas:
            if orig_aa in forbidden:
                forbidden[orig_aa] |= forbidden_aas
            else:
                forbidden[orig_aa] = forbidden_aas

    return forbidden


def _aa_forbidden_to_position_forbidden(
    aa_forbidden: dict[str, set[str]],
    vhh: "VHHSequence",
) -> dict[int, set[str]]:
    """Convert amino-acid-level forbidden substitutions to position-level.

    Parameters
    ----------
    aa_forbidden:
        Dict mapping original amino acid (single letter) to a set of
        forbidden replacement amino acids.
    vhh:
        The VHH sequence whose IMGT positions are used for expansion.

    Returns
    -------
    Dict mapping IMGT position (int) -> set of forbidden replacement AA codes.
    """
    position_forbidden: dict[int, set[str]] = {}
    for pos, aa in vhh.imgt_numbered.items():
        if aa in aa_forbidden:
            position_forbidden[pos] = aa_forbidden[aa]
    return position_forbidden


def tab_mutations(humanness_scorer, stability_scorer):
    st.header("🎯 Mutation Selection")
    if st.session_state.vhh_seq is None:
        st.info("Please analyze a sequence first (Tab 1).")
        return

    vhh = st.session_state.vhh_seq

    # --- Off-limit position selection via interactive sequence selector ---
    st.subheader("Off-Limit Mutation Positions")

    # Initialize off-limit region toggles in session state
    if "off_limit_regions" not in st.session_state:
        st.session_state.off_limit_regions = {"CDR1": True, "CDR2": True, "CDR3": True,
                                               "FR1": False, "FR2": False, "FR3": False, "FR4": False}

    # --- CSV upload for per-amino-acid forbidden substitutions ---
    # Parsed first so that forbidden-substitution markers appear on the
    # interactive selector below.
    st.subheader("Upload Forbidden Substitutions (CSV)")
    st.caption(
        "Upload a CSV where **column 1** is a single-letter amino acid code "
        "(A, R, K, etc.) and **column 2** lists the amino acids that are **not "
        "allowed** to replace the amino acid in column 1 anywhere in the VHH "
        "sequence (e.g. `VIL` or `V,I,L`). For example, a row `A,VIL` means "
        "that any position containing **A** cannot be mutated to **V**, **I**, or **L**."
    )

    col_upload, col_template = st.columns([3, 1])
    with col_template:
        # Provide a downloadable template
        template_csv = "original_aa,forbidden_replacements\nA,VIL\nC,FGHI\n"
        st.download_button(
            "📄 Template CSV",
            template_csv.encode(),
            "forbidden_substitutions_template.csv",
            "text/csv",
            help="Download a template CSV to fill in your forbidden substitutions.",
        )

    aa_forbidden: dict = {}
    forbidden_substitutions: dict = {}
    with col_upload:
        uploaded_csv = st.file_uploader(
            "Upload forbidden substitutions CSV",
            type=["csv"],
            key="forbidden_csv_upload",
        )
        if uploaded_csv is not None:
            aa_forbidden = _parse_off_limit_csv(uploaded_csv)
            if aa_forbidden:
                # Convert AA-level rules to position-level for downstream use
                forbidden_substitutions = _aa_forbidden_to_position_forbidden(aa_forbidden, vhh)
                st.success(
                    f"Loaded forbidden substitution rules for **{len(aa_forbidden)}** amino acid(s), "
                    f"affecting **{len(forbidden_substitutions)}** positions."
                )
                with st.expander("View loaded forbidden substitutions"):
                    rows = []
                    for orig_aa in sorted(aa_forbidden.keys()):
                        rows.append({
                            "Original AA": orig_aa,
                            "Forbidden Replacements": ", ".join(sorted(aa_forbidden[orig_aa])),
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.warning("Could not parse any forbidden substitutions from the uploaded CSV.")

    # --- Region toggles and interactive sequence selector kept adjacent ---
    st.markdown("---")
    st.subheader("Interactive Sequence Selector")
    st.caption(
        "Click individual residues (or click-and-drag) in the sequence below to "
        "toggle them as off-limits.  Use the region checkboxes for bulk selection. "
        "CDR regions are off-limits by default."
    )

    st.markdown("**Toggle regions off-limits:**")
    region_cols = st.columns(7)
    region_names = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
    for i, region_name in enumerate(region_names):
        is_cdr = region_name.startswith("CDR")
        with region_cols[i]:
            st.session_state.off_limit_regions[region_name] = st.checkbox(
                region_name,
                value=st.session_state.off_limit_regions.get(region_name, is_cdr),
                key=f"region_toggle_{region_name}",
            )

    # Build initial off-limit positions from toggled regions
    region_off_limit_positions = set()
    for region_name, is_off in st.session_state.off_limit_regions.items():
        if is_off:
            start, end = IMGT_REGIONS[region_name]
            for p in range(start, end + 1):
                if p in vhh.imgt_numbered:
                    region_off_limit_positions.add(p)

    st.markdown(
        "**Click residues below** to toggle off-limits (darkened = off-limit). "
        "Click-and-drag to select a range."
    )
    # Use a key that incorporates the region toggle state so the component
    # re-initialises when the user toggles an entire region.  Build a
    # deterministic string key (hash() can produce negatives / vary across runs).
    region_key_suffix = "_".join(sorted(
        k for k, v in st.session_state.off_limit_regions.items() if v
    )) or "none"
    selected_positions = sequence_selector(
        sequence=vhh.sequence,
        imgt_numbered=vhh.imgt_numbered,
        off_limit_positions=region_off_limit_positions,
        forbidden_substitutions=forbidden_substitutions,
        key=f"seq_selector_{region_key_suffix}",
    )

    # The component returns None on first render before user interaction;
    # in that case fall back to the region-derived defaults.
    if selected_positions is not None:
        off_limit_positions = set(selected_positions)
    else:
        off_limit_positions = region_off_limit_positions

    # Summary
    n_off = len(off_limit_positions)
    n_mutable = vhh.length - n_off
    n_forbidden = len(forbidden_substitutions)
    col1, col2, col3 = st.columns(3)
    col1.metric("Off-limit positions", n_off)
    col2.metric("Mutable positions", n_mutable)
    col3.metric("Positions with restricted AAs", n_forbidden)

    # --- Rank mutations ---
    hw = st.session_state.get("humanness_weight", 0.6)
    engine = MutationEngine(humanness_scorer, stability_scorer, w_humanness=hw, w_stability=1 - hw)

    if st.button("Rank Mutations", type="primary"):
        with st.spinner("Ranking mutations..."):
            try:
                df = engine.rank_single_mutations(
                    vhh,
                    off_limits=off_limit_positions,
                    forbidden_substitutions=forbidden_substitutions,
                )
                st.session_state.ranked_mutations = df
            except Exception as e:
                st.error(f"Error ranking mutations: {e}")

    if st.session_state.ranked_mutations is not None:
        df = st.session_state.ranked_mutations
        if len(df) == 0:
            st.info("No beneficial mutations found with the current settings.")
            return
        st.subheader(f"Top Mutations ({len(df)} candidates)")
        st.dataframe(df, use_container_width=True)

        n_mut = st.session_state.get("n_mutations", 5)
        max_var = st.session_state.get("max_variants", 1000)

        if st.button("Generate Library", type="primary"):
            with st.spinner(f"Generating library (up to {max_var} variants)..."):
                try:
                    library = engine.generate_library(vhh, df, n_mutations=n_mut, max_variants=max_var)
                    st.session_state.library = library
                    st.success(f"Library generated: {len(library)} variants.")
                except Exception as e:
                    st.error(f"Error generating library: {e}")


def tab_library(viz):
    st.header("📚 Library Results")
    if st.session_state.library is None:
        st.info("Generate a library first (Tab 2).")
        return

    lib = st.session_state.library
    st.metric("Total Variants", len(lib))

    st.subheader("Variant Library")
    st.dataframe(lib, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        csv_bytes = lib.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv_bytes, "library.csv", "text/csv")
    with col2:
        fasta_lines = []
        for _, row in lib.iterrows():
            fasta_lines.append(f">{row.get('variant_id','variant')}")
            fasta_lines.append(row["aa_sequence"])
        fasta_bytes = "\n".join(fasta_lines).encode()
        st.download_button("⬇️ Download FASTA", fasta_bytes, "library.fasta", "text/plain")

    if st.session_state.vhh_seq is not None and len(lib) > 0:
        st.subheader("Top 10 Variant Alignments")
        for i, row in lib.head(10).iterrows():
            with st.expander(f"{row.get('variant_id', 'VAR')} | combined={row['combined_score']:.3f} | {row['mutations']}"):
                orig = st.session_state.vhh_seq
                mut_seq = row["aa_sequence"]
                mut_info = {}
                for j, (a, b) in enumerate(zip(orig.sequence, mut_seq)):
                    if a != b:
                        mut_info[j] = "humanness"
                html = viz.render_alignment(orig, mut_seq, mut_info)
                st.components.v1.html(html, height=100)


def tab_construct(optimizer, tag_manager):
    st.header("🔧 Construct Builder")
    if st.session_state.vhh_seq is None:
        st.info("Please analyze a sequence first (Tab 1).")
        return

    vhh = st.session_state.vhh_seq
    available_tags = tag_manager.get_available_tags()
    tag_options = ["None"] + list(available_tags.keys())

    col1, col2 = st.columns(2)
    with col1:
        n_tag = st.selectbox("N-terminal Tag", tag_options, key="n_tag_select")
    with col2:
        c_tag = st.selectbox("C-terminal Tag", tag_options, key="c_tag_select")

    linker = st.text_input("Linker Sequence", value="GSGSGS")
    host = st.session_state.get("host", "e_coli")
    strategy = st.session_state.get("strategy", "most_frequent")

    if st.button("Build Construct", type="primary"):
        with st.spinner("Optimizing codons and building construct..."):
            try:
                opt_result = optimizer.optimize(vhh.sequence, host=host, strategy=strategy)
                dna = opt_result["dna_sequence"]
                construct = tag_manager.build_construct(
                    vhh.sequence, dna,
                    n_tag=None if n_tag == "None" else n_tag,
                    c_tag=None if c_tag == "None" else c_tag,
                    linker=linker,
                )
                construct["codon_opt"] = opt_result
                st.session_state.construct = construct
            except Exception as e:
                st.error(f"Error building construct: {e}")

    if st.session_state.construct:
        c = st.session_state.construct
        st.subheader("Construct Schematic")
        st.code(c["schematic"])
        st.subheader("Amino Acid Sequence")
        st.code(c["aa_construct"])
        st.subheader("DNA Sequence")
        st.code(c["dna_construct"] if c["dna_construct"] else "(DNA encoding not available for all tags)")
        if "codon_opt" in c:
            co = c["codon_opt"]
            col1, col2 = st.columns(2)
            col1.metric("GC Content", f"{co['gc_content']*100:.1f}%")
            col2.metric("CAI", f"{co['cai']:.3f}")
            if co["warnings"]:
                for w in co["warnings"]:
                    st.warning(w)
            if co["flagged_sites"]:
                for fs in co["flagged_sites"]:
                    st.error(fs)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Download AA", c["aa_construct"].encode(), "construct.fasta", "text/plain")
        with col2:
            st.download_button("⬇️ Download DNA", c["dna_construct"].encode(), "construct_dna.fasta", "text/plain")


def tab_history():
    st.header("📁 Session History")
    sessions_dir = Path("sessions")
    if not sessions_dir.exists():
        st.info("No sessions directory found.")
        return

    session_files = list(sessions_dir.glob("*.json"))
    if not session_files:
        st.info("No saved sessions found.")
        return

    st.write(f"Found {len(session_files)} session(s).")
    selected = st.selectbox("Select session to view:", [f.name for f in session_files])
    if selected and st.button("Load Session"):
        import json
        with open(sessions_dir / selected) as f:
            data = json.load(f)
        st.json(data)

    if st.session_state.library is not None or st.session_state.vhh_seq is not None:
        if st.button("💾 Save Current Session"):
            lm = st.session_state.library_manager
            save_data = {}
            if st.session_state.humanness_scores:
                save_data["humanness_scores"] = st.session_state.humanness_scores
            if st.session_state.stability_scores:
                save_data["stability_scores"] = st.session_state.stability_scores
            if st.session_state.library is not None:
                save_data["library"] = st.session_state.library
            try:
                paths = lm.save_session(save_data)
                st.success(f"Session saved: {paths}")
            except Exception as e:
                st.error(f"Error saving session: {e}")


def main():
    init_state()
    humanness_scorer, stability_scorer = load_scorers()
    optimizer = CodonOptimizer()
    tag_manager = TagManager()
    viz = SequenceVisualizer()

    hw, n_mut, max_var, host, strategy = sidebar()

    tabs = st.tabs(["🔬 Input & Analysis", "🎯 Mutation Selection", "📚 Library Results", "🔧 Construct Builder", "📁 Session History"])
    with tabs[0]:
        tab_input(humanness_scorer, stability_scorer, viz)
    with tabs[1]:
        tab_mutations(humanness_scorer, stability_scorer)
    with tabs[2]:
        tab_library(viz)
    with tabs[3]:
        tab_construct(optimizer, tag_manager)
    with tabs[4]:
        tab_history()


if __name__ == "__main__":
    main()

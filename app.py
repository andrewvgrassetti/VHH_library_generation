import streamlit as st
import pandas as pd
import io
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vhh_library.sequence import VHHSequence, IMGT_REGIONS
from vhh_library.humanness import HumAnnotator
from vhh_library.stability import StabilityScorer
from vhh_library.mutation_engine import MutationEngine
from vhh_library.codon_optimizer import CodonOptimizer
from vhh_library.tags import TagManager
from vhh_library.library_manager import LibraryManager
from vhh_library.visualization import SequenceVisualizer
from vhh_library.components import sequence_selector
from vhh_library.barcodes import BarcodeGenerator
from vhh_library.developability import (
    PTMLiabilityScorer,
    ClearanceRiskScorer,
    SurfaceHydrophobicityScorer,
)
from vhh_library.orthogonal_scoring import (
    HumanStringContentScorer,
    ConsensusStabilityScorer,
    NanoMeltStabilityScorer,
    _NANOMELT_TM_MIN,
    _NANOMELT_TM_MAX,
)

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
    ptm = PTMLiabilityScorer()
    clr = ClearanceRiskScorer()
    shyd = SurfaceHydrophobicityScorer()
    hsc = HumanStringContentScorer()
    cons = ConsensusStabilityScorer()
    # NanoMelt is optional — gracefully skip if package is not installed
    nm = NanoMeltStabilityScorer()
    if not nm.is_available:
        nm = None
    return h, s, ptm, clr, shyd, hsc, cons, nm


def init_state():
    defaults = {
        "vhh_seq": None,
        "humanness_scores": None,
        "stability_scores": None,
        "ptm_scores": None,
        "clearance_scores": None,
        "hydrophobicity_scores": None,
        "orthogonal_humanness_scores": None,
        "orthogonal_stability_scores": None,
        "nanomelt_scores": None,
        "ranked_mutations": None,
        "library": None,
        "library_manager": LibraryManager(),
        "construct": None,
        "constructs": None,
        "barcoded_library": None,
        "barcode_reference": None,
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
    st.sidebar.caption(
        "Enable/disable each metric and set its weight.  "
        "Weights are normalised automatically so they sum to 1."
    )

    # --- Metric toggles and weight sliders ---
    _METRICS = [
        ("humanness", "Humanness", True, 0.35),
        ("stability", "Stability", True, 0.25),
        ("ptm_liability", "PTM Liability", False, 0.15),
        ("clearance_risk", "Clearance Risk", False, 0.15),
        ("surface_hydrophobicity", "Surface Hydrophobicity", False, 0.10),
    ]
    enabled_metrics: dict[str, bool] = {}
    raw_weights: dict[str, float] = {}
    for key, label, default_on, default_w in _METRICS:
        col_en, col_w = st.sidebar.columns([1, 2])
        with col_en:
            enabled = st.checkbox(label, value=default_on, key=f"enable_{key}")
        with col_w:
            w = st.slider(
                f"{label} weight",
                0.0, 1.0, default_w, step=0.05,
                key=f"weight_{key}",
                disabled=not enabled,
                label_visibility="collapsed",
            )
        enabled_metrics[key] = enabled
        raw_weights[key] = w if enabled else 0.0

    # Show normalised weights
    total_w = sum(raw_weights.values())
    if total_w > 0:
        norm = {k: v / total_w for k, v in raw_weights.items()}
    else:
        norm = raw_weights
    weight_summary = "  \n".join(
        f"**{label}**: {norm[key]:.0%}" if enabled_metrics[key] else f"~~{label}~~: off"
        for key, label, _, _ in _METRICS
    )
    st.sidebar.markdown(weight_summary)

    st.sidebar.subheader("Library Parameters")
    min_mut = st.sidebar.slider("Min Mutations per Variant", 1, 20, 1, key="min_mutations")
    n_mut = st.sidebar.slider("Max Mutations per Variant", 1, 20, 5, key="n_mutations")
    if min_mut > n_mut:
        st.sidebar.warning("Min mutations exceeds max mutations — min will be clamped to max.")
    max_var = st.sidebar.number_input("Max Variants", 100, 10000, 1000, step=100, key="max_variants")

    st.sidebar.subheader("Sampling Strategy")
    sampling_strategy = st.sidebar.selectbox(
        "Strategy",
        ["Auto", "Random Sampling", "Iterative Refinement"],
        index=0,
        key="sampling_strategy",
        help=(
            "Auto: exhaustive for small spaces, random for medium, iterative for very large (>1M combinations). "
            "Random Sampling: always uses random sampling. "
            "Iterative Refinement: anchor-and-explore strategy."
        ),
    )
    anchor_threshold = st.sidebar.slider(
        "Anchor frequency threshold",
        0.1, 1.0, 0.6, step=0.05,
        key="anchor_threshold",
        disabled=(sampling_strategy == "Random Sampling"),
        help="Fraction of top-quartile variants a position-mutation must appear in to be soft-locked as anchor.",
    )
    max_rounds = st.sidebar.number_input(
        "Max refinement rounds",
        1, 10, 5, step=1,
        key="max_rounds",
        disabled=(sampling_strategy == "Random Sampling"),
        help="Maximum number of anchor-and-explore refinement rounds.",
    )

    st.sidebar.subheader("Expression System")
    host = st.sidebar.selectbox("Host", ["e_coli", "s_cerevisiae", "p_pastoris", "h_sapiens"], key="host")
    strategy = st.sidebar.selectbox("Codon Strategy", ["most_frequent", "harmonized", "gc_balanced"], key="strategy")

    return raw_weights, enabled_metrics, min_mut, n_mut, max_var, host, strategy


def tab_input(humanness_scorer, stability_scorer, ptm_scorer, clearance_scorer,
              hydrophobicity_scorer, hsc_scorer, consensus_scorer, nanomelt_scorer, viz):
    st.header("🔬 Input & Analysis")
    seq_input = st.text_area("Paste VHH amino acid sequence:", value=SAMPLE_VHH, height=100)

    if st.button("Analyze Sequence", type="primary"):
        try:
            vhh = VHHSequence(seq_input.strip())
            st.session_state.vhh_seq = vhh
            st.session_state.humanness_scores = humanness_scorer.score(vhh)
            st.session_state.stability_scores = stability_scorer.score(vhh)
            st.session_state.ptm_scores = ptm_scorer.score(vhh)
            st.session_state.clearance_scores = clearance_scorer.score(vhh)
            st.session_state.hydrophobicity_scores = hydrophobicity_scorer.score(vhh)
            st.session_state.orthogonal_humanness_scores = hsc_scorer.score(vhh)
            st.session_state.orthogonal_stability_scores = consensus_scorer.score(vhh)
            if nanomelt_scorer is not None:
                try:
                    st.session_state.nanomelt_scores = nanomelt_scorer.score(vhh)
                except Exception:
                    st.session_state.nanomelt_scores = None
            else:
                st.session_state.nanomelt_scores = None
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

    # --- Developability Metrics ---
    ptm = st.session_state.ptm_scores
    clr = st.session_state.clearance_scores
    shyd = st.session_state.hydrophobicity_scores
    if ptm and clr and shyd:
        st.subheader("Developability Metrics")
        dev_html = ""
        dev_html += viz.render_score_bar(ptm["composite_score"], "PTM Liability (higher = fewer liabilities)", "#E91E63")
        dev_html += viz.render_score_bar(clr["composite_score"], f"Clearance Risk (pI={clr['pI']:.1f})", "#FF5722")
        dev_html += viz.render_score_bar(shyd["composite_score"], "Surface Hydrophobicity (higher = fewer patches)", "#795548")
        st.components.v1.html(dev_html, height=200)

        col1, col2, col3 = st.columns(3)
        col1.metric("PTM Motifs Found", len(ptm.get("hits", [])))
        col2.metric("pI Deviation", f"{clr.get('pI_deviation', 0):.2f}")
        col3.metric("Hydrophobic Patches", shyd.get("n_patches", 0))

        all_warnings = ptm.get("warnings", []) + clr.get("warnings", []) + shyd.get("warnings", [])
        if all_warnings:
            with st.expander("Developability Warnings"):
                for w in all_warnings:
                    st.warning(w)

    # --- Orthogonal Scoring (Cross-Validation) ---
    orth_h = st.session_state.orthogonal_humanness_scores
    orth_s = st.session_state.orthogonal_stability_scores
    nm = st.session_state.nanomelt_scores
    if orth_h and orth_s:
        st.subheader("Orthogonal Scoring (Cross-Validation)")
        st.caption(
            "Independent scoring methods for cross-validating the primary scores. "
            "**Human String Content** uses k-mer peptide matching against human germlines. "
            "**Consensus Stability** scores framework positions against VHH germline consensus. "
            + ("**NanoMelt Tm** predicts apparent melting temperature using ESM-based embeddings." if nm else "")
        )
        orth_html = ""
        orth_html += viz.render_score_bar(
            orth_h["composite_score"],
            f"Human String Content ({orth_h['matched_kmers']}/{orth_h['total_kmers']} k-mers matched)",
            "#7B1FA2",
        )
        orth_html += viz.render_score_bar(
            orth_s["composite_score"],
            f"Consensus Stability ({orth_s['consensus_matches']}/{orth_s['positions_evaluated']} positions matched)",
            "#00695C",
        )
        if nm:
            orth_html += viz.render_score_bar(
                nm["composite_score"],
                f"NanoMelt Tm (predicted {nm['predicted_tm']:.1f} °C, normalised score)",
                "#F57C00",
            )
        bar_height = 150 if nm else 100
        st.components.v1.html(orth_html, height=bar_height)

        if nm:
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Primary Humanness", f"{hs['composite_score']:.3f}")
            col2.metric("Orthogonal Humanness (HSC)", f"{orth_h['composite_score']:.3f}")
            col3.metric("Primary Stability", f"{ss['composite_score']:.3f}")
            col4.metric("Orthogonal Stability (Consensus)", f"{orth_s['composite_score']:.3f}")
            col5.metric("NanoMelt Tm (°C)", f"{nm['predicted_tm']:.1f}")
            col6.metric("NanoMelt Score (normalised)", f"{nm['composite_score']:.3f}")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Primary Humanness", f"{hs['composite_score']:.3f}")
            col2.metric("Orthogonal Humanness (HSC)", f"{orth_h['composite_score']:.3f}")
            col3.metric("Primary Stability", f"{ss['composite_score']:.3f}")
            col4.metric("Orthogonal Stability (Consensus)", f"{orth_s['composite_score']:.3f}")


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

    # --- Excluded target amino acids (global) ---
    st.subheader("Excluded Target Amino Acids")
    st.caption(
        "Select amino acids that should **never** be introduced by any mutation. "
        "By default, **Cysteine (C)** is excluded to avoid introducing unintended "
        "disulfide bonds."
    )
    _ALL_AAS = list("ACDEFGHIKLMNPQRSTVWY")
    excluded_target_aas = set(st.multiselect(
        "Amino acids to exclude as mutation targets",
        options=_ALL_AAS,
        default=["C"],
        key="excluded_target_aas",
        help="Mutations to any of these amino acids will be globally blocked.",
    ))
    if excluded_target_aas:
        st.info(f"Globally excluded target amino acids: **{', '.join(sorted(excluded_target_aas))}**")

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
    weights = {}
    enabled = {}
    for key in MutationEngine.METRIC_NAMES:
        weights[key] = st.session_state.get(f"weight_{key}", 0.0)
        enabled[key] = st.session_state.get(f"enable_{key}", key in ("humanness", "stability"))
    engine = MutationEngine(
        humanness_scorer, stability_scorer,
        weights=weights, enabled_metrics=enabled,
    )

    if st.button("Rank Mutations", type="primary"):
        with st.spinner("Ranking mutations..."):
            try:
                df = engine.rank_single_mutations(
                    vhh,
                    off_limits=off_limit_positions,
                    forbidden_substitutions=forbidden_substitutions,
                    excluded_target_aas=excluded_target_aas,
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
        min_mut = st.session_state.get("min_mutations", 1)
        max_var = st.session_state.get("max_variants", 1000)

        if st.button("Generate Library", type="primary"):
            with st.spinner(f"Generating library (up to {max_var} variants)..."):
                try:
                    # Map UI strategy name to engine strategy key
                    _strategy_map = {
                        "Auto": "auto",
                        "Random Sampling": "random",
                        "Iterative Refinement": "iterative",
                    }
                    ui_strategy = st.session_state.get("sampling_strategy", "Auto")
                    eng_strategy = _strategy_map.get(ui_strategy, "auto")
                    anchor_thr = st.session_state.get("anchor_threshold", 0.6)
                    max_rnd = int(st.session_state.get("max_rounds", 5))
                    library = engine.generate_library(
                        vhh, df, n_mutations=n_mut,
                        max_variants=max_var, min_mutations=min_mut,
                        strategy=eng_strategy,
                        anchor_threshold=anchor_thr,
                        max_rounds=max_rnd,
                    )
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

    # --- Distribution plots ---
    if len(lib) > 0 and "humanness_score" in lib.columns and "stability_score" in lib.columns:
        st.subheader("Score Distributions")

        orig_h = None
        orig_s = None
        if st.session_state.humanness_scores is not None:
            orig_h = st.session_state.humanness_scores.get("composite_score")
        if st.session_state.stability_scores is not None:
            orig_s = st.session_state.stability_scores.get("composite_score")

        # Collect all available score columns and their original values
        _SCORE_COLS = [
            ("humanness_score", "Humanness", "#4CAF50", orig_h),
            ("stability_score", "Stability", "#2196F3", orig_s),
        ]
        if "ptm_liability_score" in lib.columns and st.session_state.ptm_scores:
            _SCORE_COLS.append((
                "ptm_liability_score", "PTM Liability", "#E91E63",
                st.session_state.ptm_scores.get("composite_score"),
            ))
        if "clearance_risk_score" in lib.columns and st.session_state.clearance_scores:
            _SCORE_COLS.append((
                "clearance_risk_score", "Clearance Risk", "#FF5722",
                st.session_state.clearance_scores.get("composite_score"),
            ))
        if "surface_hydrophobicity_score" in lib.columns and st.session_state.hydrophobicity_scores:
            _SCORE_COLS.append((
                "surface_hydrophobicity_score", "Surface Hydrophobicity", "#795548",
                st.session_state.hydrophobicity_scores.get("composite_score"),
            ))
        if "orthogonal_humanness_score" in lib.columns and st.session_state.orthogonal_humanness_scores:
            _SCORE_COLS.append((
                "orthogonal_humanness_score", "Orthogonal Humanness (HSC)", "#7B1FA2",
                st.session_state.orthogonal_humanness_scores.get("composite_score"),
            ))
        if "orthogonal_stability_score" in lib.columns and st.session_state.orthogonal_stability_scores:
            _SCORE_COLS.append((
                "orthogonal_stability_score", "Orthogonal Stability (Consensus)", "#00695C",
                st.session_state.orthogonal_stability_scores.get("composite_score"),
            ))
        if "nanomelt_tm_score" in lib.columns and st.session_state.nanomelt_scores:
            _SCORE_COLS.append((
                "nanomelt_tm_score", "NanoMelt Tm Score", "#F57C00",
                st.session_state.nanomelt_scores.get("composite_score"),
            ))

        n_plots = len(_SCORE_COLS)
        n_cols = min(n_plots, 3)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

        for idx, (col_name, label, color, orig_val) in enumerate(_SCORE_COLS):
            ax = axes[idx // n_cols][idx % n_cols]
            ax.hist(lib[col_name], bins=30, color=color, alpha=0.7, edgecolor="white", label="Variants")
            if orig_val is not None:
                ax.axvline(orig_val, color="#C62828", linewidth=2, linestyle="--",
                           label=f"Original ({orig_val:.3f})")
            ax.set_xlabel(f"{label} Score")
            ax.set_ylabel("Count")
            ax.set_title(f"{label} Distribution")
            ax.legend(fontsize=8)

        # Hide unused subplot axes
        for idx in range(n_plots, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # --- Orthogonal Correlation Plots ---
        _has_orth_h = "orthogonal_humanness_score" in lib.columns
        _has_orth_s = "orthogonal_stability_score" in lib.columns
        _has_nm = "nanomelt_tm_score" in lib.columns and lib["nanomelt_tm_score"].notna().any()
        n_corr = sum([_has_orth_h, _has_orth_s, _has_nm])
        if n_corr > 0:
            st.subheader("Orthogonal Score Correlation")
            st.caption(
                "Scatter plots comparing primary scores against orthogonal scores "
                "across all library variants.  Strong correlation validates that "
                "both independent scoring methods agree."
            )
            # Stability uses 2 subplots (jittered scatter + hexbin); humanness uses 1; NanoMelt uses 1
            n_cols = (_has_orth_h * 1) + (_has_orth_s * 2) + (_has_nm * 1)
            fig_corr, ax_corr_flat = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), squeeze=False)
            col_idx = 0
            if _has_orth_h:
                ax = ax_corr_flat[0][col_idx]
                ax.scatter(lib["humanness_score"], lib["orthogonal_humanness_score"],
                           alpha=0.4, s=12, color="#7B1FA2")
                ax.set_xlabel("Primary Humanness Score")
                ax.set_ylabel("Orthogonal Humanness (HSC)")
                ax.set_title("Humanness Correlation")
                # Add Spearman rank correlation coefficient
                if len(lib) > 1:
                    rho, pval = spearmanr(lib["humanness_score"], lib["orthogonal_humanness_score"])
                    ax.annotate(f"ρ = {rho:.3f} (p={pval:.2e})", xy=(0.05, 0.95),
                                xycoords="axes fraction", fontsize=11,
                                verticalalignment="top",
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                col_idx += 1
            if _has_orth_s:
                # Left: jittered scatter to reveal density within discrete bands
                ax_scatter = ax_corr_flat[0][col_idx]
                jitter_x = lib["stability_score"] + np.random.default_rng(seed=42).normal(0, 0.003, len(lib))
                ax_scatter.scatter(jitter_x, lib["orthogonal_stability_score"],
                                   alpha=0.3, s=10, color="#00695C")
                ax_scatter.set_xlabel("Primary Stability Score (jittered)")
                ax_scatter.set_ylabel("Orthogonal Stability (Consensus)")
                ax_scatter.set_title("Stability Correlation (Jittered)")
                if len(lib) > 1:
                    rho, pval = spearmanr(lib["stability_score"], lib["orthogonal_stability_score"])
                    ax_scatter.annotate(f"ρ = {rho:.3f} (p={pval:.2e})", xy=(0.05, 0.95),
                                        xycoords="axes fraction", fontsize=11,
                                        verticalalignment="top",
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                col_idx += 1

                # Right: hexbin density plot for a cleaner view of banded distributions
                ax_hex = ax_corr_flat[0][col_idx]
                hb = ax_hex.hexbin(lib["stability_score"], lib["orthogonal_stability_score"],
                                   gridsize=25, cmap="YlGnBu", mincnt=1)
                ax_hex.set_xlabel("Primary Stability Score")
                ax_hex.set_ylabel("Orthogonal Stability (Consensus)")
                ax_hex.set_title("Stability Correlation (Density)")
                fig_corr.colorbar(hb, ax=ax_hex, label="Count")
                col_idx += 1

            if _has_nm:
                nm_idx = lib["nanomelt_tm_score"].notna()
                ax_nm = ax_corr_flat[0][col_idx]
                ax_nm.scatter(lib.loc[nm_idx, "stability_score"],
                              lib.loc[nm_idx, "nanomelt_tm_score"],
                              alpha=0.4, s=10, color="#F57C00")
                ax_nm.set_xlabel("Primary Stability Score")
                ax_nm.set_ylabel("NanoMelt Tm Score (normalised)")
                ax_nm.set_title("Stability vs. NanoMelt Tm")
                if nm_idx.sum() > 1:
                    rho, pval = spearmanr(
                        lib.loc[nm_idx, "stability_score"],
                        lib.loc[nm_idx, "nanomelt_tm_score"],
                    )
                    ax_nm.annotate(f"ρ = {rho:.3f} (p={pval:.2e})", xy=(0.05, 0.95),
                                   xycoords="axes fraction", fontsize=11,
                                   verticalalignment="top",
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                # Add predicted Tm axis on the right for interpretability
                if "predicted_tm" in lib.columns and lib["predicted_tm"].notna().any():
                    ax_nm2 = ax_nm.twinx()
                    ax_nm2.set_ylabel("Predicted Tm (°C)", color="#F57C00")
                    ax_nm2.tick_params(axis="y", labelcolor="#F57C00")
                    y_lim = ax_nm.get_ylim()
                    # Scale back to Tm °C using the same normalisation formula
                    ax_nm2.set_ylim(
                        y_lim[0] * (_NANOMELT_TM_MAX - _NANOMELT_TM_MIN) + _NANOMELT_TM_MIN,
                        y_lim[1] * (_NANOMELT_TM_MAX - _NANOMELT_TM_MIN) + _NANOMELT_TM_MIN,
                    )
                col_idx += 1

            fig_corr.tight_layout()
            st.pyplot(fig_corr)
            plt.close(fig_corr)

            # --- Sub-score Breakdown Correlation Plots ---
            _sub_score_cols = [c for c in ("aggregation_score", "charge_balance_score",
                                           "hydrophobic_core_score") if c in lib.columns]
            if _has_orth_s and _sub_score_cols:
                with st.expander("Stability Sub-score Correlations", expanded=False):
                    st.caption(
                        "Scatter plots of individual continuous stability sub-scores vs. "
                        "the orthogonal consensus stability score.  These produce smoother "
                        "correlations than the composite and reveal which biophysical "
                        "properties drive germline consensus alignment."
                    )
                    n_sub = len(_sub_score_cols)
                    fig_sub, ax_sub = plt.subplots(1, n_sub, figsize=(5 * n_sub, 4), squeeze=False)
                    _sub_labels = {
                        "aggregation_score": "Aggregation Score",
                        "charge_balance_score": "Charge Balance Score",
                        "hydrophobic_core_score": "Hydrophobic Core Score",
                    }
                    _sub_colors = {
                        "aggregation_score": "#EF6C00",
                        "charge_balance_score": "#1565C0",
                        "hydrophobic_core_score": "#558B2F",
                    }
                    for i, col_name in enumerate(_sub_score_cols):
                        ax = ax_sub[0][i]
                        ax.scatter(lib[col_name], lib["orthogonal_stability_score"],
                                   alpha=0.35, s=10, color=_sub_colors.get(col_name, "#555"))
                        ax.set_xlabel(_sub_labels.get(col_name, col_name))
                        ax.set_ylabel("Orthogonal Stability (Consensus)")
                        ax.set_title(f"{_sub_labels.get(col_name, col_name)} vs. Orthogonal")
                        if len(lib) > 1:
                            rho, pval = spearmanr(lib[col_name], lib["orthogonal_stability_score"])
                            ax.annotate(f"ρ = {rho:.3f} (p={pval:.2e})", xy=(0.05, 0.95),
                                        xycoords="axes fraction", fontsize=10,
                                        verticalalignment="top",
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                    fig_sub.tight_layout()
                    st.pyplot(fig_sub)
                    plt.close(fig_sub)

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

    if st.session_state.library is None:
        st.info("Generate a library first (Tab 2).")
        return

    lib = st.session_state.library

    # -- Barcode toggle -------------------------------------------------------
    include_barcodes = st.checkbox(
        "Include barcodes in constructs",
        value=False,
        key="construct_include_barcodes",
        help="When enabled, uses barcoded sequences from the Barcoding tab. "
             "When disabled, uses the top-N un-barcoded candidate sequences.",
    )

    if include_barcodes:
        if st.session_state.get("barcoded_library") is None:
            st.warning(
                "No barcoded library found. Please assign barcodes first "
                "in the **🧬 Barcoding** tab."
            )
            return
        source_df = st.session_state["barcoded_library"]
        seq_column = "barcoded_sequence"
        st.info(f"Using {len(source_df)} barcoded constructs from the Barcoding tab.")
    else:
        construct_top_n = st.number_input(
            "Top N candidates",
            min_value=1,
            max_value=len(lib),
            value=min(10, len(lib)),
            step=1,
            key="construct_top_n",
        )
        source_df = lib.nlargest(int(construct_top_n), "combined_score").copy().reset_index(drop=True)
        seq_column = "aa_sequence"
        st.info(f"Using top {len(source_df)} un-barcoded candidates by combined score.")

    # -- Tag and linker settings ----------------------------------------------
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

    if st.button("Build Constructs", type="primary"):
        with st.spinner("Optimizing codons and building constructs..."):
            try:
                constructs = []
                for _, row in source_df.iterrows():
                    aa_seq = row.get(seq_column, row.get("aa_sequence", ""))
                    opt_result = optimizer.optimize(aa_seq, host=host, strategy=strategy)
                    dna = opt_result["dna_sequence"]
                    construct = tag_manager.build_construct(
                        aa_seq, dna,
                        n_tag=None if n_tag == "None" else n_tag,
                        c_tag=None if c_tag == "None" else c_tag,
                        linker=linker,
                    )
                    construct["codon_opt"] = opt_result
                    construct["variant_id"] = row.get("variant_id", "")
                    construct["combined_score"] = row.get("combined_score", "")
                    if include_barcodes:
                        construct["barcode_id"] = row.get("barcode_id", "")
                        construct["barcode_peptide"] = row.get("barcode_peptide", "")
                    constructs.append(construct)
                st.session_state.constructs = constructs
                # Keep single-construct backward compat
                if constructs:
                    st.session_state.construct = constructs[0]
                st.success(f"Built {len(constructs)} constructs.")
            except Exception as e:
                st.error(f"Error building constructs: {e}")

    if st.session_state.constructs:
        constructs = st.session_state.constructs

        # -- Summary table ----------------------------------------------------
        st.subheader("Construct Summary")
        summary_rows = []
        for c in constructs:
            co = c.get("codon_opt", {})
            row_data = {
                "variant_id": c.get("variant_id", ""),
                "score": c.get("combined_score", ""),
                "aa_length": len(c["aa_construct"]),
                "gc_content": f"{co.get('gc_content', 0) * 100:.1f}%",
                "cai": f"{co.get('cai', 0):.3f}",
            }
            if include_barcodes:
                row_data["barcode_id"] = c.get("barcode_id", "")
            summary_rows.append(row_data)
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        # -- Detailed view for first construct --------------------------------
        st.subheader("Construct Detail (first construct)")
        c = constructs[0]
        st.markdown("**Schematic:**")
        st.code(c["schematic"])
        st.markdown("**Amino Acid Sequence:**")
        st.code(c["aa_construct"])
        st.markdown("**DNA Sequence:**")
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

        # -- Downloads --------------------------------------------------------
        st.subheader("Download")
        # Build combined FASTA for all constructs
        aa_fasta_lines = []
        dna_fasta_lines = []
        for c in constructs:
            vid = c.get("variant_id", "construct")
            bc_id = c.get("barcode_id", "")
            header = f">{vid}"
            if bc_id:
                header += f" | {bc_id}"
            aa_fasta_lines.append(header)
            aa_fasta_lines.append(c["aa_construct"])
            dna_fasta_lines.append(header)
            dna_fasta_lines.append(c["dna_construct"] if c["dna_construct"] else "")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download All AA (FASTA)",
                "\n".join(aa_fasta_lines).encode(),
                "constructs_aa.fasta",
                "text/plain",
            )
        with col2:
            st.download_button(
                "⬇️ Download All DNA (FASTA)",
                "\n".join(dna_fasta_lines).encode(),
                "constructs_dna.fasta",
                "text/plain",
            )


def tab_barcoding():
    st.header("🧬 Barcoding")
    st.markdown(
        "Assign unique trypsin-cleavable peptide barcodes to the top variants for "
        "co-transfection multiplexed expression screening by LC-MS/MS."
    )

    if st.session_state.library is None:
        st.info("Generate a library first (Tab 2).")
        return

    lib = st.session_state.library

    enable_barcoding = st.checkbox("Enable barcoding", value=False, key="enable_barcoding")
    if not enable_barcoding:
        st.info("Check **Enable barcoding** above to assign barcodes to top candidates.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        top_n = st.number_input(
            "Top N candidates to barcode",
            min_value=1, max_value=len(lib), value=min(100, len(lib)),
            step=1, key="barcode_top_n",
        )
    with col2:
        linker = st.text_input("Linker sequence", value="GGS", key="barcode_linker")
    with col3:
        c_terminal_tail = st.text_input(
            "C-terminal protective tail",
            value="",
            key="barcode_c_tail",
            help="2-3 amino acids appended after the barcode to protect against "
                 "C-terminal endopeptidase cleavage. Must not contain K or R.",
        )

    if st.button("Assign Barcodes", type="primary"):
        with st.spinner("Assigning barcodes and building reference table..."):
            try:
                generator = BarcodeGenerator()
                barcoded = generator.assign_barcodes(
                    lib, top_n=int(top_n), linker=linker,
                    c_terminal_tail=c_terminal_tail,
                )
                st.session_state["barcoded_library"] = barcoded
                ref_table = generator.generate_barcode_reference(barcoded)
                st.session_state["barcode_reference"] = ref_table
                st.success(f"Barcodes assigned to {len(barcoded)} variants.")
            except Exception as e:
                st.error(f"Error assigning barcodes: {e}")

    if st.session_state.get("barcoded_library") is not None:
        barcoded = st.session_state["barcoded_library"]
        ref_table = st.session_state.get("barcode_reference")

        st.subheader("Barcode Assignment Table")
        display_cols = [
            c for c in [
                "variant_id", "combined_score", "mutations",
                "barcode_id", "barcode_peptide", "barcoded_sequence", "barcode_tryptic_peptide",
                "barcode_source",
            ] if c in barcoded.columns
        ]
        st.dataframe(barcoded[display_cols], use_container_width=True)

        if ref_table is not None and len(ref_table) > 0:
            st.subheader("Barcode Reference Table (for MS method setup)")
            st.dataframe(ref_table, use_container_width=True)

            # -- Biophysical distribution plots --
            st.subheader("Barcode Biophysical Distributions")
            st.markdown(
                "These plots characterise the assigned barcode peptides to help "
                "assess expected separation on a reversed-phase column feeding "
                "into electrospray ionisation (ESI). Wider, more uniform "
                "distributions indicate better chromatographic spread."
            )
            try:
                generator = BarcodeGenerator()
                fig = generator.plot_barcode_distributions(ref_table)
                st.pyplot(fig)
            except Exception as plot_err:
                st.warning(f"Could not render distribution plots: {plot_err}")

            # -- Source summary --
            if "source" in ref_table.columns:
                counts = ref_table["source"].value_counts()
                st.subheader("Barcode Source Summary")
                for src, cnt in counts.items():
                    label = src.replace("_", " ").title() if src else "Unknown"
                    st.metric(label, cnt)

        st.subheader("Download")
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                generator = BarcodeGenerator()
                fasta_str = generator.generate_barcoded_fasta(barcoded)
                st.download_button(
                    "⬇️ Barcoded FASTA",
                    fasta_str.encode(),
                    "barcoded_library.fasta",
                    "text/plain",
                )
            except Exception:
                pass
        with col2:
            if ref_table is not None and len(ref_table) > 0:
                st.download_button(
                    "⬇️ Barcode Reference CSV",
                    ref_table.to_csv(index=False).encode(),
                    "barcode_reference.csv",
                    "text/csv",
                )
        with col3:
            st.download_button(
                "⬇️ Combined Library CSV",
                barcoded.to_csv(index=False).encode(),
                "barcoded_library.csv",
                "text/csv",
            )


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
            if st.session_state.ptm_scores:
                save_data["ptm_scores"] = st.session_state.ptm_scores
            if st.session_state.clearance_scores:
                save_data["clearance_scores"] = st.session_state.clearance_scores
            if st.session_state.hydrophobicity_scores:
                save_data["hydrophobicity_scores"] = st.session_state.hydrophobicity_scores
            if st.session_state.orthogonal_humanness_scores:
                save_data["orthogonal_humanness_scores"] = st.session_state.orthogonal_humanness_scores
            if st.session_state.orthogonal_stability_scores:
                save_data["orthogonal_stability_scores"] = st.session_state.orthogonal_stability_scores
            if st.session_state.nanomelt_scores:
                save_data["nanomelt_scores"] = st.session_state.nanomelt_scores
            if st.session_state.library is not None:
                save_data["library"] = st.session_state.library
            try:
                paths = lm.save_session(save_data)
                st.success(f"Session saved: {paths}")
            except Exception as e:
                st.error(f"Error saving session: {e}")


def main():
    init_state()
    humanness_scorer, stability_scorer, ptm_scorer, clearance_scorer, hydrophobicity_scorer, hsc_scorer, consensus_scorer, nanomelt_scorer = load_scorers()
    optimizer = CodonOptimizer()
    tag_manager = TagManager()
    viz = SequenceVisualizer()

    raw_weights, enabled_metrics, min_mut, n_mut, max_var, host, strategy = sidebar()

    tabs = st.tabs(["🔬 Input & Analysis", "🎯 Mutation Selection", "📚 Library Results", "🧬 Barcoding", "🔧 Construct Builder", "📁 Session History"])
    with tabs[0]:
        tab_input(humanness_scorer, stability_scorer, ptm_scorer, clearance_scorer,
                  hydrophobicity_scorer, hsc_scorer, consensus_scorer, nanomelt_scorer, viz)
    with tabs[1]:
        tab_mutations(humanness_scorer, stability_scorer)
    with tabs[2]:
        tab_library(viz)
    with tabs[3]:
        tab_barcoding()
    with tabs[4]:
        tab_construct(optimizer, tag_manager)
    with tabs[5]:
        tab_history()


if __name__ == "__main__":
    main()

# VHH Biosimilar Library Generator

A computational tool for designing humanized and stability-optimized VHH (nanobody) variant libraries.

## Features

- **IMGT Numbering**: Automatic mapping of VHH sequences to IMGT positions with FR/CDR region identification
- **Humanness Scoring**: Framework comparison against human VH germline position-frequency matrices
- **Stability Analysis**: VHH hallmark detection, disulfide scoring, pI/charge calculation, aggregation prediction
- **Orthogonal Scoring**: Independent cross-validation via Human String Content (k-mer) humanness and VHH consensus stability scoring
- **NanoMelt Integration**: Continuous thermostability prediction (predicted Tm in °C) using ESM-based nanobody embeddings — installed by default
- **ESM-2 PLL Rescoring**: Pseudo-log-likelihood rescoring of top library candidates using the ESM-2 protein language model — toggle on/off in the sidebar
- **Mutation Engine**: Ranked single mutations + combinatorial library generation (up to 10,000 variants)
- **Codon Optimization**: E. coli, S. cerevisiae, P. pastoris with most-frequent, harmonized, and GC-balanced strategies
- **Construct Builder**: N/C-terminal tag attachment (6xHis, HA, Myc, FLAG, Aga2p, pIII_pelB) with linker support
- **Session Management**: JSON/CSV/FASTA export, session persistence and reloading
- **Streamlit UI**: Interactive 6-tab web application

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### NanoMelt and ESM-2

NanoMelt (thermostability prediction) and ESM-2 PLL (pseudo-log-likelihood rescoring)
are installed by default. Both require PyTorch and download ESM model weights on first
use (~several hundred MB).

NanoMelt scores are automatically computed during sequence analysis and library
generation. ESM-2 PLL rescoring can be enabled or disabled via the sidebar toggle
in the Streamlit UI.

## Usage

### Web Application

```bash
streamlit run app.py
```

### Python API

```python
from vhh_library import VHHSequence, HumAnnotator, StabilityScorer, MutationEngine

seq = VHHSequence("QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMG...")
h = HumAnnotator()
s = StabilityScorer()
print(h.score(seq)["composite_score"])
print(s.score(seq)["composite_score"])
```

### Running Tests

```bash
pytest tests/ -v
```

## Modules

| Module | Description |
|--------|-------------|
| `sequence.py` | VHHSequence class, IMGT numbering, region extraction |
| `humanness.py` | HumAnnotator: germline comparison, position frequency scoring |
| `stability.py` | StabilityScorer: hydrophobicity, pI, disulfide, VHH hallmarks |
| `mutation_engine.py` | MutationEngine: single-mutation ranking, combinatorial library generation |
| `codon_optimizer.py` | CodonOptimizer: host-specific DNA sequence design |
| `tags.py` | TagManager: construct assembly with N/C-terminal tags |
| `library_manager.py` | LibraryManager: session management, CSV/FASTA export |
| `visualization.py` | SequenceVisualizer: HTML alignment, region track, score bars |
| `utils.py` | Shared constants, codon table, utility functions |

## Future Integration Points

- **AbNatiV**: Deep learning-based humanness scoring (Marks & Deane, 2022)
- **TNP**: Target-specific nanobody optimization pipeline

## References

1. Spinelli, S. et al. (2001). *The crystal structure of a llama heavy chain variable domain*. Natural Structural Biology.
2. Lefranc, M.-P. et al. (2003). *IMGT unique numbering for immunoglobulin and T cell receptor variable domains*. Dev Comp Immunol.
3. Mitchell, L.S. & Colwell, L.J. (2018). *Comparative analysis of nanobody sequence and structure data*. Proteins.
4. Vincke, C. et al. (2009). *General strategy to humanize a camelid single-domain antibody*. J Biol Chem.
5. Ewert, S. et al. (2003). *Biophysical properties of human antibody variable domains*. J Mol Biol.
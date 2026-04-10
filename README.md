# VHH Biosimilar Library Generator

A tool for generating humanized VHH (nanobody) variant libraries with sequence scoring and codon optimization.

## Features

- **IMGT Numbering**: Maps VHH sequences to IMGT positions and FR/CDR regions
- **Humanness Scoring**: Position-frequency comparison against human VH germlines
- **Stability Scoring**: VHH hallmark detection, disulfide scoring, pI calculation, hydrophobicity
- **Additional Scoring**: Human String Content (k-mer) humanness and VHH consensus stability scores
- **NanoMelt Integration**: Predicted Tm via ESM-based embeddings — install `nanomelt` to enable
- **Mutation Engine**: Single-mutation ranking and combinatorial library generation (up to 10K variants)
- **Codon Optimization**: Host-specific strategies for E. coli, S. cerevisiae, and P. pastoris
- **Construct Builder**: N/C-terminal tag attachment with linker support
- **Session Management**: JSON/CSV/FASTA export and session reloading
- **Streamlit UI**: 6-tab web interface

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### Optional: NanoMelt thermostability prediction

NanoMelt predicts melting temperature (Tm) using ESM-based embeddings.
It requires PyTorch and downloads model weights on first use (~several hundred MB).

```bash
pip install nanomelt
```

Once installed, NanoMelt scores are included in sequence analysis and library generation.

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
# rules-diversity

Maximizing **pattern diversity** on an \(n\times n\) torus under symmetric rule matrices \(R\) via:
- **Stage 1 (Exact)**: DFS + Transfer matrix (trace \(T^n\)).
- **Stage 2 (Approx)**: Row enumeration + `TransferOp` (GPU/CPU) + Lanczos/Power/Hutch++.
- **NSGA-II (simplified)** genetic search over rule bits with **canonical labeling**.

## Install
```bash
git clone <your_repo_url>
cd rules-diversity
pip install -e .

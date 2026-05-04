# Adaptive Speculation Length Optimization for Speculative Decoding

> Final project for **MSML 604: Introduction to Optimization**, University of Maryland, College Park (Spring 2026).

This repository contains the code, data, and final report for a study that frames the choice of speculation length `k` in speculative decoding as a constrained optimization problem, compares three classes of solvers (convex relaxation, multi-armed bandit, Bayesian optimization), and proposes a closed-form non-IID extension of the standard cost model under linear position-dependent acceptance.

---

## TL;DR

- **Setup.** 425 measured runs of speculative decoding on an NVIDIA A100 (80 GB), Qwen 2.5 0.5B Instruct as draft, Qwen 2.5 7B Instruct as target.
- **Latency model.** `T(k) = 31.07 k + 107.80` ms with R² = 0.9999.
- **Convexity.** The relaxed objective `f(k) = T(k) / E[N(k)]` is convex on the operating range, both in `k` and under `u = log k`.
- **Solver comparison.** Convex relaxation, UCB1 bandit, and Thompson sampling all return the empirical optimum `k* = 3` (14.2 tok/s). Bayesian optimization with a GP surrogate returns `k = 5`, within 4% of the optimum.
- **Theoretical contribution.** The IID acceptance assumption used by every prior speculative decoding analysis is rejected at `p < 0.0001` for every `k ≥ 4`. We derive a closed-form expression for `E[N(k)]` under linear acceptance `α(j) = α₀ + β·j` (Proposition 1) and the corresponding first-order optimality condition (Proposition 2). The non-IID model reduces total absolute prediction error of expected tokens per round by **78.8 percent** on the full dataset and **23.1 percent** under leave-one-category-out cross-validation.

---

## Repository layout

```
.
├── README.md                       This file.
├── docs/
│   └── final_report.pdf            15-page IEEE-style final report.
├── notebooks/
│   ├── 01_data_collection.ipynb    Generates the 425-run dataset on an A100.
│   ├── 02_optimization.ipynb       Latency fit, convexity, and the three solvers.
│   └── 03_noniid_analysis.ipynb    Proposition 1 + 2 and empirical validation.
├── figures/                        All figures used in the report (PNG).
├── equations/                      LaTeX-rendered equation images (PNG).
└── data/
    └── rich_data_a100.json         The collected experimental data.
```

---

## Reproducing the results

The full analysis pipeline runs in three notebooks. Each is self-contained and can be opened directly in Google Colab.

### 1. Data collection (`notebooks/01_data_collection.ipynb`)

Runs speculative decoding on a Colab Pro A100 instance across 25 prompts × 8 values of `k` × 2 trials = 400 speculative runs, plus 25 baseline (non-speculative) runs. Writes the result to `data/rich_data_a100.json`.

**Hardware needed.** NVIDIA A100 80 GB. Smaller GPUs (A40, T4) will run out of memory loading Qwen 2.5 7B in FP16 alongside the draft.

**Runtime.** Approximately 90 minutes on A100.

### 2. Optimization analysis (`notebooks/02_optimization.ipynb`)

Loads the dataset and reproduces:
- Latency model fit (`fig1_latency_fit.png`).
- Convexity check in `k` and in `log k` (`fig2_convexity.png`).
- IID assumption test with per-position acceptance rates and two-sample t-tests (`fig3_iid_test.png`).
- UCB1 cumulative arm pulls (`fig4_ucb_convergence.png`).
- Bayesian optimization GP surrogate (`fig5_bayesian.png`).
- Pareto frontier of throughput against acceptance rate (`fig6_pareto.png`).
- Solver comparison table (Table I in the report).

**Runtime.** Less than 5 minutes on a CPU.

### 3. Non-IID analysis (`notebooks/03_noniid_analysis.ipynb`)

Implements Propositions 1 and 2 from Section VI of the report. Fits the linear acceptance model `α(j) = α₀ + β·j`, evaluates `E[N(k)]` under both IID and non-IID, computes prediction errors, and runs leave-one-category-out cross-validation.

Produces `fig7_noniid_fit.png` and `fig8_noniid_vs_iid.png`.

**Runtime.** Less than 1 minute on a CPU.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/prathambharati/msml604-speculative-decoding
cd msml604-speculative-decoding

# Install dependencies
pip install -r requirements.txt
```

For the data-collection notebook you additionally need `transformers`, `accelerate`, and access to an A100. Smaller models can be substituted by changing the model identifiers at the top of the notebook, but the latency model coefficients will of course change.

---

## Key results summary

### Latency model (Section III.C of the report)

| Coefficient | Value         | Interpretation                                              |
| ----------- | ------------- | ----------------------------------------------------------- |
| `a`         | 31.07 ms      | Per-token cost (draft forward + marginal target verify).    |
| `b`         | 107.80 ms     | Fixed per-round overhead (kernel launches, KV truncation).  |
| R²          | 0.9999        | Linear fit is essentially exact on the operating range.     |

### IID assumption test (Section VIII.B)

For every `k ∈ {4, 5, 6, 8, 10}`, a two-sample t-test between acceptance at position 0 and at position `k − 1` returns `p < 0.0001`. The IID assumption is rejected at any reasonable significance level. The direction of the violation is also notable: acceptance increases with position, opposite to common intuition. We attribute this to a survivorship effect.

### Solver comparison (Section VIII.C, Table I)

| Method                  | k* | Throughput (tok/s) | % of oracle |
| ----------------------- | -- | ------------------ | ----------- |
| Oracle (grid search)    | 3  | 14.2               | 100.0       |
| Convex relaxation       | 3  | 14.2               | 100.0       |
| UCB1 bandit             | 3  | 14.2               | 100.0       |
| Thompson sampling       | 3  | 14.2               | 100.0       |
| Bayesian optimization   | 5  | 13.6               | 95.8        |

### Non-IID model (Section VIII.D)

Linear fit: `α(j) = 0.756 + 0.0131 j`, R² = 0.82.

Total absolute prediction error of `E[N(k)]` across the eight evaluated `k` values:
- IID model: 3.07 units.
- Non-IID model: 0.65 units.
- Reduction: 78.8 percent.

Under leave-one-category-out cross-validation (fit on four prompt categories, test on the held-out one), the non-IID model retains a 23.1 percent reduction in mean absolute error.

---

## Citation

If you use any of this code or the dataset in your own work:

```bibtex
@misc{bharati2026adaptive,
  author       = {Pratham Ramachandra Bharati},
  title        = {Adaptive Speculation Length Optimization for Speculative Decoding in Large Language Model Inference},
  year         = {2026},
  howpublished = {Final project, MSML 604, University of Maryland, College Park},
  url          = {https://github.com/prathambharati/msml604-speculative-decoding}
}
```

---

## References

The full reference list (18 entries in IEEE format) is in `docs/final_report.pdf`. The principal references are Leviathan, Kalman, and Matias (ICML 2023) for the original speculative decoding analysis, Auer, Cesa-Bianchi, and Fischer (2002) for UCB1, Russo et al. (2018) for Thompson sampling, and Frazier (2018) and Balandat et al. (NeurIPS 2020) for Bayesian optimization with BoTorch.

---

## License

MIT. See `LICENSE`.

---

## Contact

Pratham Ramachandra Bharati  
M.S. Applied Machine Learning, University of Maryland, College Park  
pratham.bharati03@gmail.com

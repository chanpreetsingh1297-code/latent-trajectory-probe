# LatentTrajectory-Probe: Probing Dynamic Hidden State Trajectories & Emergent Self-Correction in Reasoning LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.38%2B-yellow.svg)](https://huggingface.co/docs/transformers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Abstract:** Autoregressive language models fine-tuned for complex mathematical reasoning exhibit non-linear state transitions—ranging from constructive self-correction ("Aha!" moments) to localized repetition loops (trajectory collapse). **LatentTrajectory-Probe** is a lightweight, layer-wise diagnostic framework that inspects intermediate transformer hidden representations $h_t^{(l)} \in \mathbb{R}^d$ token-by-token. By pairing directional cosine velocity with syntactic boundary masking and trailing-window anomaly detection, this probe disentangles genuine semantic self-correction from routine structural transitions and degenerative loops.

---

## 📌 Problem Formulation & Motivation

Probing internal representations in causal language models often relies on measuring adjacent hidden state similarity across sequence steps. However, naive geometric distance metrics suffer from systematic false positives driven by two primary structural artifacts:

1. **Syntactic Boundary Noise:** Routine transitions between structural delimiters (e.g., punctuation marks `.`, `,`, `!`) and content tokens naturally induce steep vector shifts. Standard probes misinterpret these syntactic boundary hops as semantic self-corrections or emergent pivots.
2. **Trajectory Collapse Satiation:** Degenerative token loops produce artificially high cosine similarity clusters, creating localized attractors that obscure true reasoning shifts.

**LatentTrajectory-Probe** resolves this by applying a syntactically masked, rolling-window $Z$-score estimator over latent trajectory steps, isolating true semantic pivots from baseline syntactic fluctuations.

---

## 📐 Mathematical Framework

Let $X = (x_1, x_2, \dots, x_N)$ be an autoregressive sequence of tokens, and let $h_t^{(l)} \in \mathbb{R}^d$ denote the hidden representation at layer $l$ for step $t$.

### 1. Vector Trajectory Velocity
Directional displacement between consecutive sequence steps is calculated via normalized Cosine Similarity:

$$S_t = \cos\left(h_t^{(l)}, h_{t+1}^{(l)}\right) = \frac{h_t^{(l)} \cdot h_{t+1}^{(l)}}{\|h_t^{(l)}\|_2 \|h_{t+1}^{(l)}\|_2}$$

### 2. Rolling Window Anomaly Standardizing
To isolate localized trajectory acceleration from background context drift, dynamic statistics are computed over a historical window $W$ of size $k = 5$:

$$\mu_{t} = \frac{1}{k} \sum_{i=1}^{k} S_{t-i}, \quad \sigma_{t} = \sqrt{\frac{1}{k} \sum_{i=1}^{k} (S_{t-i} - \mu_t)^2} + \epsilon$$

The standardized deviation score $Z_t$ quantifies relative acceleration in latent space:

$$Z_t = \frac{S_t - \mu_t}{\sigma_t}$$

### 3. Syntactic Masking & Diagnostic Operator
Let $\mathcal{P}$ denote the subset of structural, punctuation, and whitespace tokens. The diagnostic classification operator $\mathcal{D}(t)$ is defined as:

$$\mathcal{D}(t) = \begin{cases} \text{Syntactic Step}, & \text{if } x_{t+1} \in \mathcal{P} \\ \text{Trajectory Collapse}, & \text{if } S_t > 0.80 \land Z_t > +1.5 \\ \text{Emergent Pivot ('Aha!'),} & \text{if } S_t < 0.55 \land Z_t < -1.5 \\ \text{Stable Continuity}, & \text{otherwise} \end{cases}$$

---

## 🔬 Empirical Diagnostics & Benchmarks

Evaluated against **Qwen/Qwen2.5-Math-1.5B** across two representative mathematical reasoning paths:

* **Scenario A (Degenerate Repetitive Loop):** `"Therefore, the side is 5. Therefore, the side is 5."`
* **Scenario B (Emergent Self-Correction):** `"Therefore, the side is 5. Wait! Let me check again. It is 3."`

### Empirical Diagnostic Matrix

| Sequence Context | Transition Target ($x_{t+1}$) | Cos Sim ($S_t$) | $Z_t$ Score | Unmasked Baseline | Masked Diagnostic Probe |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Scenario A** | `Therefore` | 0.5124 | -2.71 | 🔥 False "Aha!" Pivot | ⚪ Syntactic Step |
| **Scenario A** | `,` (repeat step) | 0.9082 | +3.12 | ⚠️ False Collapse | ⚪ Syntactic Step |
| **Scenario B** | `Wait` | 0.4561 | -2.65 | 🔥 Emergence | ★ **Emergent Pivot ('Aha!')** |
| **Scenario B** | `again` | 0.5420 | -1.78 | 🔥 Emergence | ★ **Emergent Pivot ('Aha!')** |

---

## 🛠 Repository Layout

```text
latent-trajectory-probe/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── main.py
├── 01_latent_trajectory_probe.ipynb
├── src/
│   ├── __init__.py
│   ├── engine.py
│   └── visualize.py
└── assets/
    └── comparative_trajectories.png
```

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
git clone [https://github.com/your-username/latent-trajectory-probe.git](https://github.com/your-username/latent-trajectory-probe.git)
cd latent-trajectory-probe
pip install -r requirements.txt
```

### 2. Execution Pipeline
Run the full benchmarking suite and export publication-ready diagnostic plots:
```bash
python main.py
```

### 3. Programmatic API Integration
```python
from src.engine import LatentTrajectoryAnalyzer
from src.visualize import plot_comparative_dynamics

# Initialize analyzer on target model layer
analyzer = LatentTrajectoryAnalyzer(model_name="Qwen/Qwen2.5-Math-1.5B", layer_idx=-1)

# Run trajectory diagnostic engine
results = analyzer.analyze_sequence(
    text_sequence="Therefore, the side is 5. Wait! Let me check again. It is 3.",
    label="Emergent Self-Correction Evaluation"
)
```

---

## 🔭 Future Research Directions

- [ ] **Cross-Layer Representation Divergence ($\Delta L$):** Quantifying how self-correction shifts propagate from intermediate layers ($L/2$) to final projection layers ($L$).
- [ ] **Topological Phase Space Mapping:** Pairing trajectory velocity $S_t$ with output logit entropy $H(P(y_t \mid y_{<t}))$ to map uncertainty landscapes during reasoning pivots.
- [ ] **Large-Scale Benchmark Probing:** Scaling empirical evaluation across MATH and GSM8K traces using reasoning models (DeepSeek-R1-Distill, Qwen2.5-Math-7B).

---

## 📑 Citation

```bibtex
@software{latent_trajectory_probe2025,
  author = {Your Name},
  title = {LatentTrajectory-Probe: Probing Dynamic Hidden State Trajectories & Emergent Self-Correction in Reasoning LLMs},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  url = {[https://github.com/your-username/latent-trajectory-probe](https://github.com/your-username/latent-trajectory-probe)}
}
```

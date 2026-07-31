# LatentTrajectory-Probe: Diagnosing Model Laziness & Trajectory Collapse in Fine-Tuned LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.38%2B-yellow.svg)](https://huggingface.co/docs/transformers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Target Models](https://img.shields.io/badge/Target_Models-Qwen_2.5_|_VLMs-purple.svg)]()

> **Overview:** Fine-tuning open-weights models (such as Qwen 2.5) for multi-step visual and textual reasoning often introduces subtle failure modes during generation—specifically **model laziness**, repetitive token looping, and latent state stagnation. **LatentTrajectory-Probe** is a lightweight diagnostic profiler that tracks the cosine angle and similarity between hidden state vector embeddings $h_t^{(l)} \in \mathbb{R}^d$ step-by-step. By pairing raw cosine angles with syntactic boundary masking and trailing-window anomaly detection, this tool pinpoints the exact token positions where a fine-tuned model becomes lazy, gets stuck in repetitive loops, or executes genuine reasoning pivots.

---

## 💡 Motivation & Origin Story

This project was built following an analysis of the **"Visual Thinker"** research paper (and associated UCLA AI research setups on reasoning and chain-of-thought fine-tuning). A major challenge identified during fine-tuning evaluation is that models frequently exhibit **laziness** or degrade into repetitive token trajectories during long-horizon reasoning.

Standard benchmark metrics only report whether the final response is correct. They fail to reveal *where* during the generation forward pass the model lost momentum.

**LatentTrajectory-Probe** inspects the model's internal representation trajectory step-by-step using vector embeddings:
1. **Detecting Model Laziness:** When consecutive vector embeddings lock into ultra-high cosine similarity clusters ($S_t > 0.85$), indicating the model is repeating itself or procrastinating without updating its reasoning state.
2. **Filtering Out Syntactic False Positives:** Delimiters (such as punctuation or newlines) naturally induce steep cosine angle shifts. Our boundary mask prevents these structural shifts from being misdiagnosed as genuine semantic transitions.
3. **Catching Self-Correction Pivots:** Identifies sharp, non-syntactic cosine angle drops ($S_t < 0.55$) where the model dynamically corrects its trajectory ("Aha!" moments).

---

## 📐 Mathematical Framework

Let $X = (x_1, x_2, \dots, x_N)$ be an autoregressive sequence, and $h_t^{(l)} \in \mathbb{R}^d$ denote the hidden vector embedding at layer $l$ for sequence step $t$.

### 1. Vector Cosine Angle & Similarity
The directional similarity between adjacent vector embeddings is computed as:

$$S_t = \cos\left(h_t^{(l)}, h_{t+1}^{(l)}\right) = \frac{h_t^{(l)} \cdot h_{t+1}^{(l)}}{\|h_t^{(l)}\|_2 \|h_{t+1}^{(l)}\|_2}$$

The angular separation $\theta_t$ in latent representation space is:

$$\theta_t = \arccos(S_t)$$

### 2. Rolling Window Anomaly Score
To separate background context drift from localized laziness or acceleration, statistics are computed over a historical trailing window $W$ of size $k = 5$:

$$\mu_{t} = \frac{1}{k} \sum_{i=1}^{k} S_{t-i}, \quad \sigma_{t} = \sqrt{\frac{1}{k} \sum_{i=1}^{k} (S_{t-i} - \mu_t)^2} + \epsilon$$

The standardized deviation score $Z_t$ measures relative acceleration or trajectory stagnation:

$$Z_t = \frac{S_t - \mu_t}{\sigma_t}$$

### 3. Diagnostic Classifier Operator
Let $\mathcal{P}$ denote punctuation, white-space, and formatting tokens. The transition operator $\mathcal{D}(t)$ classifies step $t$ as:

$$\mathcal{D}(t) = \begin{cases} \text{Syntactic Delimiter}, & \text{if } x_{t+1} \in \mathcal{P} \\ \text{Repetitive Loop / Laziness}, & \text{if } S_t > 0.80 \land Z_t > +1.5 \\ \text{Emergent Pivot ('Aha!'),} & \text{if } S_t < 0.55 \land Z_t < -1.5 \\ \text{Stable Reasoning}, & \text{otherwise} \end{cases}$$

---

## 🔬 Diagnostic Output Matrix

Evaluated on **Qwen/Qwen2.5-Math-1.5B** across two representative generation paths:

* **Trace A (Model Laziness & Loop):** `"Therefore, the side is 5. Therefore, the side is 5."`
* **Trace B (Emergent Correction):** `"Therefore, the side is 5. Wait! Let me check again. It is 3."`

### Token Embedding Trajectory Comparison

| Sequence Context | Target Token ($x_{t+1}$) | Cos Sim ($S_t$) | $Z_t$ Score | Naive Baseline | Profiler Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Trace A** | `Therefore` | 0.5124 | -2.71 | 🔥 False "Aha!" Pivot | ⚪ Syntactic Step |
| **Trace A** | `,` (repeat step) | 0.9082 | +3.12 | ⚠️ False Collapse | ⚪ Syntactic Step |
| **Trace B** | `Wait` | 0.4561 | -2.65 | 🔥 Emergence | ★ **Emergent Pivot ('Aha!')** |
| **Trace B** | `again` | 0.5420 | -1.78 | 🔥 Emergence | ★ **Emergent Pivot ('Aha!')** |

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

## 🚀 Quickstart & Profiling Guide

### 1. Installation

```bash
git clone [https://github.com/your-username/latent-trajectory-probe.git](https://github.com/your-username/latent-trajectory-probe.git)
cd latent-trajectory-probe
pip install -r requirements.txt
```

### 2. Run Diagnostic Benchmark

Run the full profiler against sample reasoning traces and export trajectory plots to `assets/comparative_trajectories.png`:

```bash
python main.py
```

### 3. Profile Your Fine-Tuned Model Checkpoint

Analyze custom fine-tuned model checkpoints (e.g., Qwen 2.5) to locate where laziness or repetition occurs:

```python
from src.engine import LatentTrajectoryAnalyzer

# Load target model (e.g., Qwen 2.5 fine-tuned checkpoint)
analyzer = LatentTrajectoryAnalyzer(
    model_name="Qwen/Qwen2.5-Math-1.5B", 
    layer_idx=-1  # Inspect final hidden layer representations
)

# Run embedding cosine trajectory analysis
results = analyzer.analyze_sequence(
    text_sequence="Therefore, the side is 5. Wait! Let me check again. It is 3.",
    label="Qwen 2.5 Reasoning Diagnostics"
)

# Print step-by-step vector trajectory scores
for token, sim, z, status in zip(results["tokens"], results["cos_sims"], results["z_scores"], results["assessments"]):
    print(f"Token: {token:<12} | CosSim: {sim:.4f} | Z-Score: {z:+.2f} | Status: {status}")
```

---

## 📚 References & Acknowledgments

* Inspired by the **"Visual Thinker"** paper and research on fine-tuning reasoning models (including UCLA AI research on chain-of-thought trajectories and fine-tuning behaviors in the Qwen 2.5 series).
* Built using Hugging Face `transformers` and PyTorch.

---

## 📄 License

Distributed under the [MIT License](LICENSE).

# Chemotherapy Toxicity Risk Prediction (LSTM + Attention)

**Deep learning pipeline for predicting Grade 3+ chemotherapy-induced toxicity in breast cancer patients — interpretable, reproducible, and deployable.**

---

## Overview

This repository implements a complete, production-minded pipeline that:

- **Simulates clinically realistic patient data** (n = 2,000) for the first three chemotherapy cycles.  
- **Learns temporal risk patterns** using an LSTM with a custom attention layer.  
- **Produces interpretable risk scores** (probability + attention weights per cycle).  
- **Evaluates model performance** with discrimination, calibration, and subgroup analyses.  
- **Exposes a REST API** for real-time predictions via FastAPI.

**Primary goal:** provide clinicians with trustworthy, actionable risk estimates to support decisions such as prophylactic G‑CSF use and enhanced monitoring.

---

## Background and Motivation

Predicting severe (Grade 3+) chemotherapy toxicity early enables targeted interventions that can reduce morbidity and unplanned hospitalizations. Real-world clinical datasets are often restricted; synthetic, domain-informed data lets us:

- Prototype model architectures and evaluation strategies without privacy constraints.  
- Encode known clinical relationships (age, ECOG, ANC, regimen) so the model learns medically meaningful signals.  
- Focus on interpretability and calibration — both essential for clinical adoption.

---

## Data

### Synthetic Cohort Generation

- **Patients:** 2,000 simulated patients.  
- **Cycles:** first 3 chemotherapy cycles per patient.  
- **Key features per cycle:** `age`, `regimen_type`, `ecog_performance_status`, `current_neutrophils_10e9_L`.  
- **Target:** binary indicator — whether the patient experienced any Grade 3 toxicity across the three cycles.

**Generation logic (high level):**

- Baseline ANC sampled from a truncated normal distribution.  
- ANC declines across cycles with stochastic drops.  
- Toxicity probability is a logistic-like linear combination of risk factors: older age, poor ECOG, and low ANC increase risk. Probabilities are clipped to realistic bounds.

**Why synthetic?** Privacy-safe prototyping, reproducibility, and the ability to encode domain knowledge explicitly.

---

## Feature Engineering and Sequences

### Per-cycle features

- **Numeric:** `age`, `current_neutrophils_10e9_L`.  
- **Ordinal:** `ecog_performance_status`.  
- **Categorical:** `regimen_type` (Anthracycline-based, Taxane-based, CMF, Other).

### Derived features

- **Binary flags:** `age_gt_70`, `high_risk_regimen` (Anthracycline or Taxane).  
- **Interaction:** `age_x_regimen = age * high_risk_regimen`.  
- **Clinical bins:** `anc_risk_band` with thresholds $\([<1.5,\ 1.5\!-\!2.5,\ >2.5]\times10^9/L\)$.

### Sequence construction

- Group cycle-level rows by `patient_id` to form sequences of shape $\((T, F)\)$ where $\(T=3\)$ and $\(F\)$ is the number of features per cycle.  
- Pad sequences to fixed length $\(T=3\)$ using post-padding; use a `Masking` layer in the model to ignore padded timesteps.

---

## Model Architecture

### High-level design

- **Input:** sequence of cycle-level features, shape $\((T, F)\)$.  
- **Masking:** ignore padded timesteps.  
- **LSTM:** 64 units, `return_sequences=True` to produce per-timestep hidden states.  
- **Attention:** custom trainable attention that computes a scalar weight per timestep and returns a weighted sum of LSTM outputs plus the attention weights for interpretability.  
- **Dense head:** Dropout(0.3) → Dense(32, ReLU) → Dense(1, Sigmoid) for probability output.

### Attention math

Given LSTM outputs $\(H = [h_1, h_2, \dots, h_T]\)$ with $\(h_t \in \mathbb{R}^d\)$:

1. Compute unnormalized scores:
   $$\[
   e_t = \tanh(h_t W + b)
   \]
   where \(W \in \mathbb{R}^{d \times 1}\) and \(b \in \mathbb{R}\).$$

2. Normalize with softmax across timesteps:
   $$\[
   a_t = \frac{\exp(e_t)}{\sum_{k=1}^T \exp(e_k)}
   \]$$

3. Context vector (weighted sum):
   $$\[
   c = \sum_{t=1}^T a_t \cdot h_t
   \]$$

4. Final prediction:
   $$\[
   \hat{y} = \sigma(\text{Dense}(\text{Dropout}(c)))
   \]$$

**Interpretability:** $\(a_t\)$ are attention weights that indicate the relative importance of each cycle.

---

## Training Details

- **Loss:** binary cross-entropy.  
- **Optimizer:** Adam, learning rate $\(1\mathrm{e}{-3}\)$.  
- **Batch size:** 32.  
- **Epochs:** 20 (adjustable).  
- **Class weighting:** `{0:1, 1:2}` to upweight the minority (toxic) class and improve sensitivity.  
- **Random seeds:** set for `numpy` and `tensorflow` to improve reproducibility.

**Hyperparameters to tune (suggested):**

- LSTM units (32, 64, 128)  
- Dropout rate (0.1–0.5)  
- Learning rate (1e-4 to 1e-2)  
- Class weight ratio or focal loss alternative

---

## Evaluation

### Metrics and interpretation

- **AUROC (Area Under ROC):** discrimination across thresholds.  
- **AUPRC (Average Precision):** useful for imbalanced data; focuses on positive class performance.  
- **Accuracy:** overall correctness (less informative for imbalanced data).  
- **Recall (Sensitivity):** proportion of true toxic patients detected — **clinically critical**.  
- **Precision:** proportion of flagged patients who truly had toxicity — important to avoid unnecessary interventions.  
- **Brier score:** measures calibration (lower is better).  
- **Calibration curve and reliability diagram:** visualize predicted probability vs observed frequency.

**Formulas**

- AUROC and AUPRC are computed via ranking-based integrals (use `sklearn` implementations).  
- Binary cross-entropy loss:
  $$\[
  \mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]
  \]$$

### Example results (test set)

| Metric | Value |
|---|---:|
| **AUROC** | **0.749** |
| **AUPRC** | **0.816** |
| **Brier score** | *low (well-calibrated)* |
| **Recall** | *high (priority)* |
| **Precision** | *balanced to avoid over-alerting* |

> These values are illustrative from the synthetic experiment and indicate reasonable discrimination and strong precision-recall performance.

### Subgroup analysis

Evaluate performance in clinically relevant subgroups (e.g., `Age > 70`, `ECOG >= 2`) to detect performance degradation or bias. Report AUROC and AUPRC per subgroup and ensure sample sizes are sufficient for reliable estimates.

---

## Interpretability and Visualization

- **Attention weights per cycle:** bar charts showing \(a_1, a_2, a_3\) for a patient — clinicians can see which cycle drove the prediction.  
- **Global feature importance:** permutation importance or SHAP on aggregated features (after flattening sequences or using summary statistics) to complement attention.  
- **Calibration plots:** reliability diagrams and calibration curves to assess whether predicted probabilities match observed frequencies.  
- **Confusion matrix:** visualize trade-offs at the chosen threshold.

---

## Deployment and API

### FastAPI endpoint

- **Endpoint:** `POST /predict`  
- **Input:** JSON body with `sequence` — list of cycle dictionaries containing the features used in training. Example:

```json
{
  "sequence": [
    {"age": 72, "ecog_performance_status": 2, "current_neutrophils_10e9_L": 2.1},
    {"age": 72, "ecog_performance_status": 2, "current_neutrophils_10e9_L": 1.5},
    {"age": 72, "ecog_performance_status": 2, "current_neutrophils_10e9_L": 0.9}
  ]
}
```

- **Output:** JSON with `predicted_probability`, `predicted_label`, `interpretation`, and `attention_weights`.

### Example server snippet

```python
# load model with custom Attention
model = load_model("toxicity_lstm_attention.h5", custom_objects={"Attention": Attention})

# build attention model to return both proba and weights
attention_model = Model(model.input, [model.output, model.layers[3].output])
```

**Operational considerations**

- Validate input schema and feature order.  
- Normalize/scale inputs consistently with training preprocessing.  
- Add rate limiting, authentication, and logging for production use.  
- Consider model versioning and A/B testing when deploying updated models.

---

## Reproducibility and How to Run

### Requirements

- Python 3.8+  
- `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `fastapi`, `uvicorn`

Install:

```bash
pip install -r requirements.txt
```

### Quick start

1. **Train model**
   ```bash
   python train_pipeline.py
   ```
   This script should:
   - Generate synthetic data (or load real data if available)
   - Build sequences and preprocess
   - Train the LSTM + Attention model
   - Save the trained model to `toxicity_lstm_attention.h5`

2. **Run API**
   ```bash
   uvicorn app:app --reload
   ```

3. **Test prediction**
   - POST JSON to `/predict` using `curl` or Postman.

### File structure (recommended)

```
.
├─ data/
│  ├─ synthetic_generation.py
├─ notebooks/
│  ├─ exploratory_analysis.ipynb
├─ src/
│  ├─ model.py              # model architecture and Attention layer
│  ├─ train_pipeline.py
│  ├─ evaluate.py
│  ├─ api.py                # FastAPI app
├─ Project_workflow.png
├─ requirements.txt
├─ README.md
```

---

## Limitations and Ethical Considerations

- **Synthetic data:** results on synthetic data do not guarantee identical performance on real-world clinical data. External validation on real cohorts is required.  
- **Bias risk:** synthetic generation may not capture all demographic or clinical heterogeneity; subgroup evaluation and fairness audits are essential.  
- **Interpretability caveats:** attention weights indicate relative importance among timesteps but are not a full causal explanation. Combine attention with other interpretability tools (SHAP, permutation importance).  
- **Clinical deployment:** model outputs are decision support only — they must be integrated with clinical workflows, validated prospectively, and used alongside clinician judgment.

---

## Future Work

- Validate on multi-institutional real-world datasets.  
- Extend sequences beyond 3 cycles and incorporate time gaps between cycles.  
- Add richer features: comorbidities, lab trends, medication interactions, prior hospitalizations.  
- Explore alternative architectures: Transformer encoders, temporal convolutional networks, or hybrid models combining static and dynamic features.  
- Implement continuous monitoring and model drift detection in production.

---

## Contact and Demo

- **Interactive demo:** https://chemotoxpredict.lovable.app/  
- **Repository:** https://github.com/VorCollective/AdML-Capstone

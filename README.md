# Chemotherapy Toxicity Prediction in Breast Cancer Patients  

**Machine Learning Pipeline for Identifying High-Risk Chemotherapy Toxicity**

## Project Overview

This repository implements an **end-to-end predictive modeling pipeline** to identify breast cancer patients at elevated risk of **severe (Grade 3+ per CTCAE)** chemotherapy-induced toxicity and unplanned hospitalizations during the first three treatment cycles.

The pipeline uses **synthetic but clinically realistic data** (n=2,000 patients) simulating demographics, baseline labs, performance status, regimen type, and other oncology-relevant factors. Two classifiers are compared — **Logistic Regression** and **Random Forest** — with the final selection being a **calibrated Logistic Regression** model due to its strong interpretability, well-calibrated probabilities, and competitive performance.

**Key final performance (test set):**
- AUROC: **0.749**
- AUPRC: **0.816**
- Brier score: well-calibrated (low value)

The model is designed to support **clinical decision-making**, such as risk stratification, shared decision discussions, and selective use of prophylactic G-CSF.

## Project Links

- **Interactive Demo**: [ChemoTox Predict Web App](https://chemotoxpredict.lovable.app/)
- **Source Code**: [github.com/VorCollective/AdML-Capstone](https://github.com/VorCollective/AdML-Capstone)

## Workflow Overview

Below is the complete end-to-end flowchart of the modeling pipeline:

![Chemotherapy Toxicity Prediction Pipeline](Flowchart.png)

The diagram covers:
- Synthetic data generation
- Patient-level data aggregation (cycles 1–3)
- Feature engineering
- Preprocessing
- Model training & hyperparameter tuning
- Model comparison & selection
- Calibration
- Evaluation (discrimination, precision-recall, calibration plots)
- Clinical interpretation & deployment via FastAPI

## Detailed Pipeline Steps

### 1. Data Preparation
- Simulated cohort: **2,000 breast cancer patients**, adjuvant/neoadjuvant chemotherapy, first 3 cycles
- Features extracted: demographics, baseline labs (ANC, hemoglobin, platelets), ECOG status, comorbidities, regimen type, receptor status, stage, prophylactic G-CSF use, etc.
- Targets: binary indicators for **Grade 3+ toxicity** and **unplanned hospitalization**

### 2. Feature Engineering
Domain-informed transformations include:
- Binary flags: `age_gt_70`, `high_risk_regimen` (anthracycline- or taxane-based)
- Interaction: `age × high_risk_regimen`
- Clinical categorization: `anc_risk_band` (<1.5, 1.5–2.5, >2.5 × 10⁹/L)

Example code:
```
python
patient_df['age_gt_70'] = (patient_df['age'] > 70).astype(int)
patient_df['high_risk_regimen'] = patient_df['regimen_type'].isin(['Anthracycline-based', 'Taxane-based']).astype(int)
patient_df['age_x_regimen'] = patient_df['age'] * patient_df['high_risk_regimen']

patient_df['anc_risk_band'] = pd.cut(
    patient_df['baseline_neutrophils_10e9_L'],
    bins=[-np.inf, 1.5, 2.5, np.inf],
    labels=['<1.5', '1.5-2.5', '>2.5']
)
```
### 3. Preprocessing Pipeline
```
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, numeric_features),
    ('cat', cat_pipe, categorical_features),
    ('bin', 'passthrough', binary_features)
])
```
### 4. Model Development

Algorithms: Logistic Regression (saga solver, balanced weights, tuned C) vs. Random Forest (balanced weights, tuned n_estimators & max_depth)
Optimization: GridSearchCV with stratified cross-validation
Final choice: Calibrated Logistic Regression (best balance of interpretability + calibration)

### 5. Model Evaluation

Custom evaluation function:
```
def evaluate_model(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        'auroc': roc_auc_score(y_test, proba),
        'auprc': average_precision_score(y_test, proba),
        'brier': brier_score_loss(y_test, proba),
        'accuracy': accuracy_score(y_test, pred),
        'sensitivity': recall_score(y_test, pred),
        'precision': precision_score(y_test, pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
    }
```
Visual diagnostics: ROC curve, Precision-Recall curve, calibration plot, permutation importance.
### 6. Deployment (FastAPI)

Real-time prediction endpoint: POST /predict

Example request (JSON):
```
{
  "age": 72,
  "ecog_performance_status": 2,
  "comorbidities_count": 3,
  "regimen_type": "Anthracycline-based",
  "prophylactic_gcsf": 1,
  "baseline_neutrophils_10e9_L": 2.1,
  "baseline_hemoglobin_g_dl": 11.5,
  "baseline_platelets_10e9_L": 250,
  "cumulative_dose_mg_m2": 120.0,
  "delta_neutrophils": -1.2,
  "menopausal_status": "Post",
  "er_positive": 1,
  "pr_positive": 0,
  "her2_positive": 0,
  "stage": "II",
  "anc_risk_band": "1.5-2.5",
  "age_gt_70": 1,
  "high_risk_regimen": 1,
  "age_x_regimen": 72
}
```
Example response:
```
{
  "predicted_probability": 0.68,
  "predicted_label": 1,
  "interpretation": "High risk – consider prophylactic G-CSF and enhanced monitoring"
}
```
Installation & Quick Start
```
git clone https://github.com/VorCollective/AdML-Capstone.git
cd AdML-Capstone

pip install -r requirements.txt

# Train and save the model pipeline
python train_pipeline.py

# Launch the FastAPI server
uvicorn app:app --reload
```

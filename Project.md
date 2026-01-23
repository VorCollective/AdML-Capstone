# 30-Day Hospital Readmission Risk Prediction for Cancer Patients

## Statement of the Problem

Cancer patients frequently experience unplanned hospital readmissions within 30 days of discharge, with rates often ranging from **20% to 30%** in oncology populations — significantly higher than the general inpatient population. These readmissions are commonly driven by:

- Treatment-related complications (e.g., chemotherapy-induced neutropenia leading to febrile episodes, severe nausea/vomiting/dehydration, mucositis, pain crises)
- Disease progression or recurrence
- Infections and sepsis
- Poor symptom control
- Medication side effects or polypharmacy issues
- Inadequate post-discharge support and follow-up
- Comorbid conditions exacerbated by cancer or its treatment

Unplanned readmissions impose a substantial burden on healthcare systems, including:

- Increased healthcare costs (readmissions are expensive and often penalized under value-based care models)
- Prolonged hospital stays and resource strain on oncology wards
- Reduced quality of life for patients due to repeated hospitalizations
- Emotional and financial stress on patients and families
- Missed opportunities for timely outpatient or palliative care interventions

Despite growing emphasis on reducing avoidable readmissions in oncology care pathways, hospitals and oncology teams still lack reliable, actionable tools to prospectively identify high-risk patients at the time of discharge.

Current risk-stratification approaches often rely on subjective clinical judgment or simple rule-based scores that do not adequately capture the complex, multifactorial nature of readmission risk in cancer patients (e.g., interaction between recent chemotherapy, lab abnormalities, prior utilization patterns, discharge disposition, and social determinants).

**Healthcare Predictive Analytics – Oncology-Specific**  
Forecasting unplanned 30-day readmission risk in patients with cancer using anonymized clinical data.

**Version**: 3.1 (Cancer Focus with Real Data Guidance)  
**Date**: January 2026  
**Author**: L.L  
**Location**: Nairobi, Kenya

---

## 1. Project Overview

### Objective

Develop machine learning models to predict the probability of **unplanned hospital readmission within 30 days** after discharge for **cancer patients**.

This enables oncology teams to:
- Identify high-risk patients early
- Implement targeted interventions (enhanced discharge planning, toxicity monitoring, palliative care coordination, follow-up scheduling)

### Clinical & Business Value

- Cancer patients experience elevated 30-day readmission rates (~20–30%) due to:
  - Treatment complications (neutropenia, infections, dehydration, pain crises)
  - Disease progression
  - Comorbidities and polypharmacy
- Predictive risk stratification → fewer avoidable readmissions → improved quality of life, reduced healthcare costs, better resource allocation in oncology settings.
- Aligns with global oncology quality improvement initiatives.

### Key Challenge

No large-scale, fully public, patient-level dataset exists exclusively for cancer readmission prediction.  
**Best publicly accessible option**: **MIMIC-IV**, filtered to cancer/neoplasm-related admissions.

> **Important Disclaimer**  
> This project is for **educational and research purposes only**.  
> Models are **not clinically validated** and must **never** be used for direct patient care without prospective multi-site validation, IRB/ethical approval, bias/fairness audits, and full regulatory compliance (e.g., Data Protection Act – Kenya).

---

## 2. Recommended Dataset: MIMIC-IV

**Medical Information Mart for Intensive Care IV**  
- **Source**: https://physionet.org/content/mimiciv/ (v2.2+ recommended)  
- **Access**: Free after completing CITI “Data or Specimens Only Research” training (~2–3 hours) + signing Data Use Agreement  
- **Size**: ~524,000 admissions (2008–2019)  
- **Cancer cohort size**: ~10,000–30,000 relevant encounters after filtering (depending on inclusion criteria)  
- **Target**: Binary unplanned 30-day readmission (engineered from admission timestamps; exclude planned readmissions where possible)  
- **Typical readmission rate in filtered cancer subset**: 20–28%

### Why MIMIC-IV for Cancer Readmission?

- Rich structured data: labs (neutrophils, CRP, albumin, creatinine), medications (chemo agents, antibiotics, opioids), procedures (central lines, biopsies, radiation), diagnoses/procedures (ICD codes), length of stay, discharge disposition, prior admissions/ED visits  
- Widely used in 2023–2025 oncology and critical care readmission studies

### Quick Cohort Filtering Example (Pandas)

```python
import pandas as pd

diagnoses = pd.read_csv('mimiciv/hosp/diagnoses_icd.csv.gz')

# Rough neoplasm filter (ICD-10: C*, D0*; ICD-9: 140–239)
cancer_mask = (
    diagnoses['icd_code'].str.startswith(('C', 'D0')) |
    ((diagnoses['icd_version'] == 9) & diagnoses['icd_code'].between('140', '239'))
)

cancer_hadm_ids = diagnoses[cancer_mask]['hadm_id'].unique()
print(f"Found {len(cancer_hadm_ids):,} cancer-related hospital admissions")
```
Fallback options if MIMIC access is delayed:

MIMIC-III (older but similar): https://physionet.org/content/mimiciii/
Synthetic/general hospital readmission datasets on Kaggle (augment with simulated cancer features)

### 3. Preprocessing & Feature Engineering (Oncology Emphasis)
High-Impact Features from Literature

| Category              | Examples of Strong Predictors                                      | Why Important in Cancer Readmission                          |
|-----------------------|--------------------------------------------------------------------|--------------------------------------------------------------|
| Prior Utilization     | Number of prior admissions / ED visits                             | Strongest overall predictor                                  |
| Laboratory Values     | Neutrophil count, hemoglobin, albumin, creatinine, CRP/ESR         | Infection, toxicity, malnutrition risk                       |
| Treatment-related     | Recent chemotherapy/radiation, medication changes                  | Chemo toxicity, post-procedure complications                 |
| Disease Status        | Metastatic status, Charlson comorbidity index                      | Advanced disease increases readmission likelihood            |
| Discharge             | Home vs facility vs palliative/hospice                             | Social support & follow-up capability                        |
| Demographics          | Age, derived performance status proxy                              | Frailty and physiological reserve                            |


Recommended Steps

Filter cohort using neoplasm ICD codes
Handle sparse labs/vitals: last observation carried forward (LOCF) per admission or median imputation
Create oncology-specific flags:
recent_chemo (≤30 days)
neutropenia_risk (low neutrophils + recent chemo)
metastatic (ICD codes indicating spread)

Address class imbalance (~20–30% positive): class weights, SMOTE, focal loss

### 4. Modeling Approaches & Expected Performance

| Model               | Strengths in Oncology Context                          | Expected AUC (Filtered MIMIC / Cancer Cohorts)      |
|---------------------|--------------------------------------------------------|-----------------------------------------------------|
| Logistic Regression | Interpretable odds ratios (chemo, labs, mets)          | 0.65 – 0.75                                         |
| XGBoost / LightGBM  | Excellent on tabular data + imbalance                  | 0.72 – 0.82 (frequent top performer)                |
| Random Forest       | Robust to missing values & non-linearity               | 0.70 – 0.78                                         |
| Neural Network      | Captures complex feature interactions                  | 0.68 – 0.80                                         |


Evaluation focus:

Primary: AUC-ROC
Secondary: Precision-Recall AUC, F1-score (at chosen threshold), calibration plots
Validation: Stratified / temporal cross-validation + hold-out set


### 5. Starter Modeling Code (XGBoost)
```
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Assume X (features), y (readmitted_30d binary) are prepared
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    random_state=42,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6
)

model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
print(f"Test AUC: {roc_auc_score(y_test, probs):.4f}")
```
### 6. Next Steps & Recommendations

Obtain MIMIC-IV access and construct cancer cohort
Add SHAP / permutation importance for explainability
Hyperparameter tuning (Optuna, Hyperopt)
Build interactive demo (Streamlit / Gradio)
Explore temporal validation and external generalizability


### 7. Limitations & Ethical Considerations

MIMIC-IV is US / ICU-centric → potential bias toward severe cases
Historical data (up to 2019) → does not reflect newer therapies (e.g., widespread immunotherapy, CAR-T)
No clinical deployment without prospective validation in diverse settings (including low- and middle-income countries)


### References

MIMIC-IV: https://physionet.org/content/mimiciv/
Recent studies: MIMIC-based oncology / readmission prediction papers (2023–2025)

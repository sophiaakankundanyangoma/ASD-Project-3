# Interpretable Machine Learning for Multi-Age Autism Spectrum Disorder Prediction
### A Pipeline with SHAP-Based Explainability

> **Authors:** Abenaitwe Allan · Akankunda Sophia Nyangoma · Augustine Beilel · Nantumbwe Dorothy · Tiko Joy · Turyasingura Kennedy  
> **Corresponding Author:** Dr. Kimera Richard  
> **Institution:** Faculty of Computing and Informatics, Mbarara University of Science and Technology (MUST), Uganda  
> **Submitted to:** PLOS Digital Health

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Grounding](#theoretical-grounding)
3. [Project Objectives](#project-objectives)
4. [Dataset Description](#dataset-description)
5. [Project Structure](#project-structure)
6. [Pipeline Summary](#pipeline-summary)
7. [Requirements](#requirements)
8. [How to Run](#how-to-run)
9. [Outputs](#outputs)
10. [License](#license)

---

## Overview

Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition affecting approximately 1 in 100 children globally, characterized by persistent deficits in social communication and restricted, repetitive patterns of behaviour (DSM-5). Despite its prevalence, formal diagnosis remains slow, costly, and reliant on specialist clinical evaluation — creating a critical gap in low- and middle-income settings such as Uganda where access to diagnostic services is limited.

This project presents an **end-to-end, interpretable machine learning pipeline** for ASD screening across three age groups: children, adolescents, and adults. The pipeline addresses two persistent methodological challenges in existing ML-based ASD research:

1. **Class imbalance**; ASD-positive cases are underrepresented relative to non-ASD cases, which can mislead classifiers and inflate standard accuracy metrics.
2. **Limited model interpretability**; black-box models limit clinical trust and adoption; clinicians require transparent, explainable predictions.

To address these challenges, the pipeline employs **Precision–Recall AUC (PR-AUC)** as the primary evaluation metric and integrates **SHapley Additive exPlanations (SHAP)** for both global and local model interpretability.

---

## Theoretical Grounding

### 1. Machine Learning for ASD Screening
Questionnaire-based ASD screening datasets particularly those derived from the Autism-Spectrum Quotient 10-item screener (AQ-10) provide structured feature matrices amenable to supervised ML classification. Ensemble methods such as Random Forest and XGBoost have consistently demonstrated strong performance on such data, attributed to their ability to capture non-linear interactions among behavioural indicators.

### 2. Class Imbalance
In medical screening datasets, the minority class (ASD-positive) is clinically the most important. Standard accuracy metrics can be misleadingly high when the classifier simply predicts the majority class. The **Precision–Recall curve and its area under the curve (PR-AUC)** more faithfully reflects minority-class detection performance than ROC-AUC under class imbalance conditions (Davis & Goadrich, 2006).

### 3. Explainable AI with SHAP
SHAP (Lundberg & Lee, 2017) is grounded in cooperative game theory and assigns each feature an importance value  a (**Shapley value**)representing its average marginal contribution to the model prediction across all possible feature subsets. SHAP satisfies desirable theoretical properties:
- **Local accuracy** — explanations are consistent with the model output
- **Missingness** — features absent from a sample have zero impact
- **Consistency** — if a model changes such that a feature has higher impact, its SHAP value does not decrease

**TreeSHAP** (Lundberg et al., 2020) extends SHAP specifically for tree-based ensembles (Random Forest, XGBoost), computing exact Shapley values in polynomial time.

### 4. Feature Engineering Decision
The composite `result` column a direct sum of A1–A10 behavioral item scores was deliberately **excluded from model training**. Including it would cause it to dominate both model predictions and SHAP explanations, masking the individual diagnostic signal of each AQ-10 screening item. Excluding it produces a more interpretable and clinically meaningful feature importance profile.

### 5. Key References
- American Psychiatric Association. *DSM-5*, 2013.
- Lundberg, S.M. & Lee, S.I. A unified approach to interpreting model predictions. *NeurIPS*, 2017.
- Lundberg, S.M. et al. From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2020.
- Davis, J. & Goadrich, M. The relationship between Precision-Recall and ROC curves. *ICML*, 2006.
- Thabtah, F. Autism Screening Adult Dataset. UCI ML Repository, 2017.
- Chen, T. & Guestrin, C. XGBoost: A scalable tree boosting system. *KDD*, 2016.
- Breiman, L. Random Forests. *Machine Learning*, 2001.

---

## Project Objectives

1. **Build a reproducible, end-to-end ML pipeline** for ASD classification using combined behavioral screening data across children, adolescents, and adults.
2. **Address class imbalance** using a Dummy Classifier baseline and PR-AUC as the primary evaluation metric to ensure minority-class performance is accurately measured.
3. **Compare four classifiers** — Dummy Baseline, Logistic Regression, Random Forest, and XGBoost — within a unified experimental framework.
4. **Integrate SHAP-based explainability** at both global (population-level) and local (individual-level) scales to provide clinically interpretable insights.
5. **Contribute a publicly available pipeline** that can be replicated, extended, and applied in resource-constrained digital health settings.

---

## Dataset Description

Three publicly available ASD behavioral screening datasets were sourced from the **UCI Machine Learning Repository**:

| Dataset | Instances | ASD Positive | Source |
|---------|-----------|-------------|--------|
| Autism Screening — Adults | 704 | ~30% | UCI ML Repository |
| Autism Screening — Children | 292 | ~50% | UCI ML Repository |
| Autism Screening — Adolescents | 104 | ~50% | UCI ML Repository |
| **Combined** | **1,100** | **~38.5%** | Merged |

**Features used for training (20 variables):**
- AQ-10 behavioral items: `A1_Score` through `A10_Score`
- Demographic: `age`, `gender`, `ethnicity`, `jundice`, `austim`, `contry_of_res`, `used_app_before`
- Contextual: `age_desc`, `relation`, `Age_Group`

> **Note:** The `result` composite score and `id` identifier column were excluded before training.

**Download datasets:**
- https://archive.ics.uci.edu/dataset/426/autism+screening+adult
- https://archive.ics.uci.edu/dataset/419/autism+screening+child
- https://archive.ics.uci.edu/dataset/420/autism+screening+adolescent

---

## Project Structure

```
ASD_Project/
│
├── data/                          # Raw dataset files (not included in repo)
│   ├── Autism_Adult_Data.csv
│   ├── Autism_Child_Data.csv
│   └── Autism_Adolescent_Data.csv
│
├── outputs/                       # All generated figures and results
│   ├── correlation_heatmap.tiff
│   ├── target_class_distribution.tiff
│   ├── hist_gender_by_ASD.tiff
│   ├── confusion_matrices.tiff
│   ├── pr_curve_comparison.tiff
│   ├── rf_feature_importance.tiff
│   ├── shap_summary.tiff
│   ├── shap_local_no_asd.tiff
│   └── shap_local_asd.tiff
│
├── main.py                        # Main pipeline script
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

---

## Pipeline Summary

The pipeline executes the following steps in sequence:

| Step | Description |
|------|-------------|
| 1 | Create output directory |
| 2 | Load and merge three age-group datasets |
| 3 | Data cleaning — handle missing values, strip whitespace |
| 4 | Label encode all categorical features |
| 4b | Leakage detection via Pearson correlation check |
| 4c | Generate correlation heatmap |
| 5 | Define features (X) and target (y); drop `result` column |
| 6 | Plot target class distribution |
| 7 | Stratified 80/20 train–test split |
| 8 | StandardScaler feature normalization |
| 9 | Train four classifiers (Dummy, Logistic Regression, Random Forest, XGBoost) |
| 10 | Generate predictions and predicted probabilities |
| 11 | Compute and compare Accuracy, Precision, Recall, F1, PR-AUC |
| 12 | Print full classification reports |
| 13 | Plot confusion matrices (RF and XGBoost) |
| 14 | Plot combined Precision–Recall curves |
| 15 | Plot Random Forest feature importance |
| 16 | SHAP TreeExplainer — global beeswarm plot |
| 16b | SHAP local waterfall — ASD-negative individual |
| 16c | SHAP local waterfall — ASD-positive individual |

---

## Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
shap
```

**Python version:** 3.8 or higher recommended.

---

## How to Run

1. **Clone or download** this repository into PyCharm or your preferred IDE.

2. **Place the datasets** in the `data/` folder:
   - `Autism_Adult_Data.csv`
   - `Autism_Child_Data.csv`
   - `Autism_Adolescent_Data.csv`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the pipeline:**
   ```bash
   python main.py
   ```

5. All figures and results will be saved automatically to the `outputs/` folder.

---

## Outputs

After running `main.py`, the following files will be generated in `outputs/`:

| File | Description |
|------|-------------|
| `correlation_heatmap.tiff` | Pearson correlation heatmap of all features |
| `target_class_distribution.tiff` | Bar chart of ASD vs non-ASD class counts |
| `hist_gender_by_ASD.tiff` | Gender distribution by ASD diagnosis |
| `confusion_matrices.tiff` | Confusion matrices for RF and XGBoost |
| `pr_curve_comparison.tiff` | Precision–Recall curves for all four models |
| `rf_feature_importance.tiff` | Random Forest mean impurity decrease feature importance |
| `shap_summary.tiff` | SHAP beeswarm global summary plot (XGBoost) |
| `shap_local_no_asd.tiff` | SHAP waterfall plot — ASD-negative individual |
| `shap_local_asd.tiff` | SHAP waterfall plot — ASD-positive individual |

---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- **Attribution** — You must give appropriate credit to the authors, provide a link to the license, and indicate if changes were made.

> Abenaitwe Allan, Akankunda Sophia Nyangoma, Augustine Beilel, Nantumbwe Dorothy, Tiko Joy, Turyasingura Kennedy & Kimera Richard (2026). *Interpretable Machine Learning for Multi-Age Autism Spectrum Disorder Prediction: A Pipeline with SHAP-Based Explainability.* Faculty of Computing and Informatics, Mbarara University of Science and Technology, Uganda.

Full license text: https://creativecommons.org/licenses/by/4.0/

---

> *This project was submitted to PLOS Digital Health for peer review. The pipeline, code, and documentation were developed as part of a Master of Science in Information Systems group project at Mbarara University of Science and Technology, Uganda, under the supervision of Dr. Kimera Richard.*

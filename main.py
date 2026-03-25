# AUTISM DATASET MODELING
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    average_precision_score, precision_recall_curve
)

import shap
import warnings
warnings.filterwarnings("ignore")

# Step 1; Creating an Output directory
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 2; Loading the datasets

adult = pd.read_csv("data/Autism_Adult_Data.csv")
child = pd.read_csv("data/Autism_Child_Data.csv")
adolescent = pd.read_csv("data/Autism_Adolescent_Data.csv")

# Add age group
adult["Age_Group"] = "Adult"
child["Age_Group"] = "Child"
adolescent["Age_Group"] = "Adolescent"

# Combine datasets
df = pd.concat([adult, child, adolescent], ignore_index=True)
print("Combined Dataset Shape:", df.shape)
print("\nDATASET OVERVIEW")
print(df.head())

# Step 3; Data cleaning

df = df.replace("?", np.nan)
df = df.dropna()

if "id" in df.columns:
    df = df.drop(columns=["id"])

df.columns = df.columns.str.strip()

# Step 4; encoding categorical features

label_encoders = {}
categorical_cols = df.select_dtypes(include=["object", "string"]).columns

for col in categorical_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Check for data leakage/ Perfect Correlation

TARGET_COL = "Class/ASD"
corr_with_target = df.corr()[TARGET_COL].sort_values(ascending=False)
leaky_features = corr_with_target[(corr_with_target == 1.0) | (corr_with_target == -1.0)]
leaky_features = leaky_features.drop(TARGET_COL, errors='ignore')

if not leaky_features.empty:
    print("\nWARNING: Perfectly correlated features with target detected!")
    print(leaky_features)
    df = df.drop(columns=leaky_features.index.tolist())
else:
    print("\nNo features perfectly correlated with target. Proceeding...")

# Step 4b; Correlation Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.tiff", dpi=300)
plt.show()

# Step 5; Features & Target
# ---------------------------------------------------------------
# DROP 'result' here — it is a composite score computed directly
# from A1-A10 and dominates SHAP values, masking the individual
# behavioural features we want to explain.
# ---------------------------------------------------------------
TARGET_COL = "Class/ASD"
X = df.drop(columns=[TARGET_COL])
X = X.drop(columns=['result'], errors='ignore')  # <-- FIX: drop before training
y = df[TARGET_COL]

print(f"\nFeatures used for training ({X.shape[1]} total):")
print(X.columns.tolist())

# Step 6; Target Distribution Plot

plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Target Class Distribution (ASD)")
plt.xlabel("ASD Diagnosis (0 = No ASD, 1 = ASD)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/target_class_distribution.tiff", dpi=300)
plt.show()

print("\nTarget class distribution:\n", y.value_counts())

# Feature Histogram: Gender vs Class/ASD Target Variable

plt.figure(figsize=(6,5))
sns.histplot(
    data=df,
    x='gender',
    hue='Class/ASD',
    multiple='dodge',
    shrink=0.8,
    palette='Set1'
)
plt.xticks([0,1], ['Female', 'Male'])
plt.title("Gender Distribution by ASD Diagnosis")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title='ASD Diagnosis', labels=['No ASD', 'ASD'])
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/hist_gender_by_ASD.tiff", dpi=300)
plt.show()

# Step 7; Train/Test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTrain samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
print(f"Train class ratio (ASD=1): {y_train.mean():.2%}, Test class ratio: {y_test.mean():.2%}")

# Step 8; Feature Scaling

numeric_cols = X.select_dtypes(include=['int64','float64']).columns
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Step 9; Model Training

# Baseline Model
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

# Logistic Regression Model
log_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])
log_pipeline.fit(X_train, y_train)

# Random Forest Model
rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
])
rf_pipeline.fit(X_train, y_train)

# XGBoost Model
xgb_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBClassifier(eval_metric="logloss", random_state=42))
])
xgb_pipeline.fit(X_train, y_train)

# Step 10; Predictions & Probabilities
baseline_preds = baseline.predict(X_test)
rf_preds       = rf_pipeline.predict(X_test)
xgb_preds      = xgb_pipeline.predict(X_test)
log_preds      = log_pipeline.predict(X_test)

baseline_probs = baseline.predict_proba(X_test)[:,1]
rf_probs       = rf_pipeline.predict_proba(X_test)[:,1]
xgb_probs      = xgb_pipeline.predict_proba(X_test)[:,1]
log_probs      = log_pipeline.predict_proba(X_test)[:,1]

# Step 11; Model Performance Comparison

def pr_auc_score(y_true, probs):
    return average_precision_score(y_true, probs)

results = pd.DataFrame({
    "Model": ["Baseline","Logistic Regression","Random Forest","XGBoost"],
    "Accuracy":[
        accuracy_score(y_test, baseline_preds),
        accuracy_score(y_test, log_preds),
        accuracy_score(y_test, rf_preds),
        accuracy_score(y_test, xgb_preds)
    ],
    "Precision":[
        precision_score(y_test, baseline_preds, zero_division=0),
        precision_score(y_test, log_preds),
        precision_score(y_test, rf_preds),
        precision_score(y_test, xgb_preds)
    ],
    "Recall":[
        recall_score(y_test, baseline_preds, zero_division=0),
        recall_score(y_test, log_preds),
        recall_score(y_test, rf_preds),
        recall_score(y_test, xgb_preds)
    ],
    "F1 Score":[
        f1_score(y_test, baseline_preds, zero_division=0),
        f1_score(y_test, log_preds),
        f1_score(y_test, rf_preds),
        f1_score(y_test, xgb_preds)
    ],
    "PR-AUC":[
        pr_auc_score(y_test, baseline_probs),
        pr_auc_score(y_test, log_probs),
        pr_auc_score(y_test, rf_probs),
        pr_auc_score(y_test, xgb_probs)
    ]
})
print("\nMODEL PERFORMANCE COMPARISON")
print(results)

# Step 12; Classification Reports For All Models

models_preds = {
    "Baseline": baseline_preds,
    "Logistic Regression": log_preds,
    "Random Forest": rf_preds,
    "XGBoost": xgb_preds
}

for name, preds in models_preds.items():
    print(f"\n--- {name} Classification Report ---")
    print(classification_report(y_test, preds, zero_division=0))

# Step 13; Confusion Matrices (RF & XGB)

fig, axes = plt.subplots(1,2, figsize=(12,4))
for ax, preds, title in zip(
    axes,
    [rf_preds, xgb_preds],
    ["Random Forest", "XGBoost"]
):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No ASD','ASD'], yticklabels=['No ASD','ASD'])
    ax.set_title(f"{title} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.tiff", dpi=300)
plt.show()

# Step 14; Precision-Recall Curves (Combined)

plt.figure(figsize=(8,6))
models = {
    "Baseline": baseline,
    "Logistic Regression": log_pipeline,
    "Random Forest": rf_pipeline,
    "XGBoost": xgb_pipeline
}
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:,1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    plt.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pr_curve_comparison.tiff", dpi=300)
plt.show()

# Step 15; Random Forest Feature Importance

rf_model = rf_pipeline.named_steps["model"]
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rf_feature_importance.tiff", dpi=300)
plt.show()

# Step 16: SHAP Explainability (XGBoost)
# 'result' is already excluded from training so X_test is clean —
# no need to drop anything here.

xgb_model = xgb_pipeline.named_steps["model"]
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_test)

# 16a. Global summary
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_summary.tiff", dpi=300)
plt.show()

# 16b. Local explanation - Person WITHOUT ASD
idx_no_asd = (y_test == 0).values.argmax()
plt.figure(figsize=(12,6))
shap.plots.waterfall(shap_values[idx_no_asd], show=False)
plt.title("Local SHAP - Person without ASD", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_local_no_asd.tiff", dpi=300)
plt.show()

# 16c. Local explanation - Person WITH ASD
idx_asd = (y_test == 1).values.argmax()
plt.figure(figsize=(12,6))
shap.plots.waterfall(shap_values[idx_asd], show=False)
plt.title("Local SHAP - Person with ASD", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_local_asd.tiff", dpi=300)
plt.show()

print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
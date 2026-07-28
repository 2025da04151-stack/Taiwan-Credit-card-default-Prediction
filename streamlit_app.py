# ------------- Declaring Libraries -----------------------
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import joblib
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# ------------- Dataset Paths --------------------------
BASE_FOLDER = os.path.dirname(__file__)
MODEL_FOLDER = os.path.join(BASE_FOLDER, "model")
DEFAULT_TESTFILE = os.path.join(BASE_FOLDER, "Test_data.csv")
TARGET_COLUMN = "default payment next month"

# ------------- Model Pickle Files ----------------------
MODEL_DUMP_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree Classifier": "decision_tree_classifier.pkl",
    "KNN Classifier": "knn_classifier.pkl",
    "Naive Bayes Classifier": "naive_bayes_classifier.pkl",
    "Random Forest(Ensemble)": "random_forest.pkl",
}

@st.cache_resource
def load_classification_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_DUMP_FILES[name]))

# --------------------------- Sidebar ----------------------------
with st.sidebar:
    
    st.subheader(" Upload Test Data file ")
    uploaded_test_data = st.file_uploader("Upload csv file only", type=["csv"])

    if uploaded_test_data is not None:
        df_test = pd.read_csv(uploaded_test_data)
        st.success(f"{uploaded_test_data.name} ({len(df_test):,} rows)")
    else:
        df_test = pd.read_csv(DEFAULT_TESTFILE)
        st.info(f"Default test data ({len(df_test):,} rows)")

    st.divider()

    st.subheader("Select Classification Model")
    selected_model = st.selectbox(
        "Choose from below:",
        list(MODEL_DUMP_FILES.keys()),
        index=4,
    )
# ---------------- Checking for Target Column Presence ---------------------
if TARGET_COLUMN not in df_test.columns:
    st.error(f"Uploaded csv file should have target column: '{TARGET_COL}'")
    st.stop()
# ----------------- Dropping multicolinary columns -------------------------------------   
df_test = df_test.drop(columns=["ID","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"])

# ----------------- Separating Target column from feature list -------------------------
X = df_test.drop(columns=[TARGET_COLUMN])
y = df_test[TARGET_COLUMN]

# ------------------------ Loading Model file --------------------------------
model = load_classification_model(selected_model)

X_test = X.values

# ----------------------- Predicting the Credit Default -----------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# --------------- Screen Design for the App ------------------------------------
st.title("Credit Card Defaulter Prediction")
st.write("""
Interactively explore **Logistic Regression,Deecision Tree, KNN, Naive Bayes, Random Forest**.
Upload test data to check the Evaluation Metrics.
""")

st.header("Model Evaluation Metrics")
st.subheader(f"{selected_model}")

eval_11, eval_12, eval_13 = st.columns(3)
eval_11.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
eval_12.metric("AUC Score", f"{roc_auc_score(y, y_proba):.4f}")
eval_13.metric("Precision", f"{precision_score(y, y_pred):.4f}")

eval_21, eval_22, eval_23 = st.columns(3)
eval_21.metric("Recall", f"{recall_score(y, y_pred):.4f}")
eval_22.metric("F1 Score", f"{f1_score(y, y_pred):.4f}")
eval_23.metric("MCC", f"{matthews_corrcoef(y, y_pred):.4f}")

st.divider()

# ----------------- Confusion Matrix + ROC Curve -----------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4.8))
    sns.heatmap(
    cm, annot=True, fmt=",d", cmap="Blues", ax=ax_cm,
    xticklabels=["Non Defaulter", "Defaulter"],
    yticklabels=["Non Defaulter", "Defaulter"],
    linewidths=0.5, annot_kws={"size": 12},
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig_cm)
    plt.close()

with col_right:
    st.markdown("**ROC Curve**")
    fpr, tpr, _ = roc_curve(y, y_proba)
    fig_roc, ax_roc = plt.subplots(figsize=(4, 3.2))
    ax_roc.plot(fpr, tpr, color="#2563eb", lw=2,
          label=f"AUC = {roc_auc_score(y, y_proba):.3f}")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig_roc)
    plt.close()

# ---------------------- Classification Report --------------------------------
st.markdown("**Classification Report**")
report = classification_report(y, y_pred, target_names=["Non Defaulter", "Defaulter"])
st.code(report, language="text")

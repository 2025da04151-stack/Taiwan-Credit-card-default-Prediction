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
DEFAULT_TESTFILE = os.path.join(BASE_FOLDER, "test_data.csv")
TARGET_COLUMN = "default payment next month"

# ------------- Model Pickle Files ----------------------
MODEL_DUMP_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree ": "decision_tree_classifier.pkl",
    "KNN Classifier": "knn_classifier.pkl",
    "Naive Bayes": "naive_bayes_classifier.pkl",
    "Random Forest": "random_forest.pkl",
}

st.cache_resource.clear()

@st.cache_resource
def load_classification_model(name):
    return joblib.load(os.path.join(MODEL_FOLDER, MODEL_DUMP_FILES[name]))

# --------------------------- Sidebar ----------------------------
with st.sidebar:
    
    st.subheader(" Upload Data file ")
    uploaded_test_data = st.file_uploader("Upload csv file only", type=["csv"])

    if uploaded_test_data is not None:
        df_test = pd.read_csv(uploaded_test_data)
        st.success(f"{uploaded_test_data.name} ({len(df_test):,} rows)")
    else:
        df_test = pd.read_csv(DEFAULT_TESTFILE)
        st.markdown(f"[Default Test Data({len(df_test):,} rows)](https://github.com/2025da04151-stack/Taiwan-Credit-card-default-Prediction/blob/main/test_data.csv)")
        st.info("The Metrics are shown based on Default test data. Upload File to get that data specific Evaluation Metrics")
    
    st.divider()

    st.subheader("Select Classification Model")
    selected_model = st.selectbox(
        "Choose from below:",
        list(MODEL_DUMP_FILES.keys()),
        index=4,
    )
# ---------------- Checking for Target Column Presence ---------------------
if TARGET_COLUMN not in df_test.columns:
    st.error(f"Uploaded csv file should have target column: '{TARGET_COLUMN}'")
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
st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Taiwan Credit Card Defaulter Prediction</h1>",
    unsafe_allow_html=True,
)
st.write("""
Interactively explore **Logistic Regression,Deecision Tree, KNN, Naive Bayes, Random Forest**.
Upload Data and Select Classification Model in left pane to check the Evaluation Metrics.
""")

st.markdown(
    """
    <style>
    /* Target the tab container buttons */
    button[data-baseweb="tab"] {
        padding: 12px 24px !important;
    }

    /* Target all text elements inside the tab buttons */
    button[data-baseweb="tab"] *, 
    button[data-baseweb="tab"] span, 
    button[data-baseweb="tab"] div {
        font-size: 50px !important;    /* Increase size as needed */
        font-weight: 700 !important;   /* Bold text */
        line-height: 1.3 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
tab_selected_model, tab_all_compare = st.tabs([
    "Selected Model Metrics", "All Models Comparison"
])

with tab_selected_model:
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
    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4.8))
        sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues", ax=ax_cm,
        xticklabels=["Non Defaulter", "Defaulter"],
        yticklabels=["Non Defaulter", "Defaulter"],
        linewidths=0.7, annot_kws={"size": 20},
        linecolor="#000000"
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close()

    with col_mid:
        st.markdown("**ROC Curve**")
        fpr, tpr, _ = roc_curve(y, y_proba)
        fig_roc, ax_roc = plt.subplots(figsize=(4, 3.2))
        ax_roc.plot(fpr, tpr, color="#2563eb", lw=2,
              label=f"AUC = {roc_auc_score(y, y_proba):.4f}")
        ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.legend(loc="lower right")
        ax_roc.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close()
    # ---------------------- Classification Report --------------------------------
    with col_right:
        st.markdown("**Classification Report**")
        report = classification_report(y, y_pred, target_names=["Non Defaulter", "Defaulter"])
        st.code(report, language="text")

with tab_all_compare:
    st.subheader(f"All Model Comparision on Data Uploaded")
    col_model = st.columns(5)
    #eval_tab = "| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |\n"
    #eval_tab += "|-------|----------|-----|-----------|--------|-----|-----|\n"
    for cols, classification_model in zip(col_model, MODEL_DUMP_FILES.keys()):
        with cols.container(border=True):
            cmodel = load_classification_model(classification_model)
            y_pred_model = cmodel.predict(X_test)
            y_proba_model = cmodel.predict_proba(X_test)[:, 1]
            #eval_tab += f"| {classification_model} | {accuracy_score(y, y_pred):.4f} | {roc_auc_score(y, y_proba):.4f} | {precision_score(y, y_pred):.4f} | {recall_score(y, y_pred):.4f} | {f1_score(y, y_pred):.4f} | {matthews_corrcoef(y, y_pred):.4f} |\n"
            #st.markdown(eval_tab)
            st.subheader(f"{classification_model}")
            st.metric("Accuracy", f"{accuracy_score(y, y_pred_model):.4f}")
            st.divider()
            st.metric("AUC Score", f"{roc_auc_score(y, y_proba_model):.4f}")
            st.divider()
            st.metric("Precision", f"{precision_score(y, y_pred_model):.4f}")
            st.divider()
            st.metric("Recall", f"{recall_score(y, y_pred_model):.4f}")
            st.divider()
            st.metric("F1 Score", f"{f1_score(y, y_pred_model):.4f}")
            st.divider()
            st.metric("MCC", f"{matthews_corrcoef(y, y_pred_model):.4f}")
            st.divider()
            
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y, y_pred_model)
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.2))
            sns.heatmap(
            cm, annot=True, fmt=",d", cmap="Blues", ax=ax_cm,
            xticklabels=["Non Defaulter", "Defaulter"],
            yticklabels=["Non Defaulter", "Defaulter"],
            linewidths=0.7, annot_kws={"size": 20},
            linecolor="#000000"
            )
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close()
            
            st.divider()

            st.markdown("**ROC Curve**")
            fpr, tpr, _ = roc_curve(y, y_proba_model)
            fig_roc, ax_roc = plt.subplots(figsize=(4, 3.2))
            ax_roc.plot(fpr, tpr, color="#2563eb", lw=2,
            label=f"AUC = {roc_auc_score(y, y_proba_model):.4f}")
            ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
            ax_roc.set_xlabel("FPR")
            ax_roc.set_ylabel("TPR")
            ax_roc.legend(loc="lower right")
            ax_roc.grid(alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig_roc)
            plt.close()


from pathlib import Path
import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from maternal_mlp import MaternalRiskMLP


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Maternal Risk Classification | Zimbabwe",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "results" / "final_models"
EVALUATION_DIR = BASE_DIR / "results" / "final_evaluation"

FEATURES = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate"
]

CLASSES = [
    "low risk",
    "mid risk",
    "high risk"
]

DISPLAY_NAMES = {
    "low risk": "Low Risk",
    "mid risk": "Medium Risk",
    "high risk": "High Risk"
}


# ============================================================
# 2. LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():

    rf_model = joblib.load(
        MODEL_DIR / "maternal_random_forest.joblib"
    )

    with open(
        MODEL_DIR / "mlp_preprocessing.json",
        "r",
        encoding="utf-8"
    ) as file:
        preprocessing = json.load(file)

    checkpoint = torch.load(
        MODEL_DIR / "maternal_mlp.pt",
        map_location="cpu",
        weights_only=True
    )

    if preprocessing["features"] != FEATURES:
        raise ValueError("MLP feature order does not match.")

    if checkpoint["classes"] != CLASSES:
        raise ValueError("MLP class labels do not match.")

    mlp_model = MaternalRiskMLP()

    mlp_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    mlp_model.eval()

    return rf_model, mlp_model, preprocessing


# ============================================================
# 3. MATERNAL RISK CLASSIFICATION FUNCTION
# ============================================================

def predict_maternal_risk(
    input_values,
    rf_model,
    mlp_model,
    preprocessing
):

    data = pd.DataFrame(
        [input_values],
        columns=FEATURES
    )

    # Apply the same heart-rate quality rule
    # used during model development.
    data.loc[
        data["HeartRate"] < 40,
        "HeartRate"
    ] = np.nan

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    rf_prediction = rf_model.predict(data)[0]

    # --------------------------------------------------------
    # DEEP LEARNING MLP
    # --------------------------------------------------------

    X = data.to_numpy(dtype=float)

    medians = np.array(
        preprocessing["imputer_medians"]
    )

    means = np.array(
        preprocessing["scaler_means"]
    )

    scales = np.array(
        preprocessing["scaler_scales"]
    )

    # Impute using saved training-data statistics.
    X = np.where(
        np.isnan(X),
        medians,
        X
    )

    # Standardise using saved training-data statistics.
    X = (X - means) / scales

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    with torch.no_grad():
        logits = mlp_model(X_tensor)

        prediction_id = int(
            torch.argmax(logits, dim=1).item()
        )

    mlp_prediction = CLASSES[prediction_id]

    return mlp_prediction, rf_prediction


# ============================================================
# 4. LOAD ARTIFACTS
# ============================================================

try:
    rf_model, mlp_model, preprocessing = load_models()

except Exception as error:
    st.error(
        "The trained model files could not be loaded. "
        "Check that the model artifacts exist."
    )
    st.exception(error)
    st.stop()


# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🩺 MaternalRisk AI")

    st.write(
        "A deep-learning research prototype for "
        "classification of maternal risk categories."
    )

    st.divider()

    st.subheader("Research Information")

    st.write("**Project:** MSc Data Science")
    st.write("**Research Context:** Zimbabwean Healthcare")
    st.write("**Primary Model:** Deep Learning MLP")
    st.write("**Comparative Model:** Random Forest")
    st.write("**Classification:** Low, Medium and High Risk")

    st.divider()

    st.warning(
        "For academic research and demonstration only. "
        "Not approved for clinical use."
    )


# ============================================================
# 6. MAIN APPLICATION HEADER
# ============================================================

st.title(
    "MATERNAL COMPLICATION RISK CLASSIFICATION "
    "AND DETECTION MODEL FOR ZIMBABWEAN HEALTHCARE"
)

st.markdown(
    "**Deep Learning-Based Maternal Health Risk Classification System**"
)

st.caption(
    "MSc Data Science Research Prototype | "
    "Multilayer Perceptron (MLP) and Random Forest"
)

st.warning(
    "IMPORTANT: This application predicts categories from "
    "a publicly available maternal health benchmark dataset. "
    "It has not been clinically validated in Zimbabwe and "
    "must not be used to diagnose complications, determine "
    "treatment, or make antenatal-care or referral decisions."
)

st.divider()


# ============================================================
# 7. APPLICATION TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "🩺 Maternal Risk Classification",
    "📊 Model Performance",
    "📚 About the Research"
])


# ============================================================
# TAB 1 — MATERNAL RISK CLASSIFICATION
# ============================================================

with tab1:

    st.header("Maternal Health Risk Classification")

    st.write(
        "Enter six maternal health indicators below. "
        "The application will compare the risk categories "
        "predicted by the Deep Learning MLP and Random Forest."
    )

    with st.form("maternal_risk_form"):

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Demographic and Blood Pressure")

            age = st.number_input(
                "Maternal Age (years)",
                min_value=0,
                max_value=110,
                value=25,
                step=1
            )

            systolic = st.number_input(
                "Systolic Blood Pressure (mmHg)",
                min_value=0,
                max_value=300,
                value=120,
                step=1
            )

            diastolic = st.number_input(
                "Diastolic Blood Pressure (mmHg)",
                min_value=0,
                max_value=200,
                value=80,
                step=1
            )

        with col2:

            st.subheader("Other Physiological Indicators")

            blood_sugar = st.number_input(
                "Blood Sugar (dataset-reported units)",
                min_value=0.0,
                max_value=50.0,
                value=7.5,
                step=0.1
            )

            body_temp = st.number_input(
                "Body Temperature (°F)",
                min_value=80.0,
                max_value=115.0,
                value=98.0,
                step=0.1
            )

            heart_rate = st.number_input(
                "Heart Rate (beats/minute)",
                min_value=0,
                max_value=220,
                value=76,
                step=1
            )

        submitted = st.form_submit_button(
            "CLASSIFY MATERNAL RISK",
            type="primary",
            use_container_width=True
        )

    if submitted:

        input_values = {
            "Age": age,
            "SystolicBP": systolic,
            "DiastolicBP": diastolic,
            "BS": blood_sugar,
            "BodyTemp": body_temp,
            "HeartRate": heart_rate
        }

        # Identify values outside the observed development
        # data range. These limits are not clinical thresholds.
        outside_range = (
            age < 10 or age > 70
            or systolic < 70 or systolic > 160
            or diastolic < 49 or diastolic > 100
            or blood_sugar < 6 or blood_sugar > 19
            or body_temp < 98 or body_temp > 103
            or heart_rate > 90
        )

        if outside_range:
            st.warning(
                "Some measurements are outside the ranges "
                "observed in the development dataset. "
                "Predictions may be unreliable."
            )

        if heart_rate < 40:
            st.info(
                "Heart rate below 40 bpm is treated as "
                "an invalid measurement and replaced "
                "using the saved training-data median."
            )

        mlp_result, rf_result = predict_maternal_risk(
            input_values,
            rf_model,
            mlp_model,
            preprocessing
        )

        st.divider()

        st.subheader("Classification Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.markdown("#### Deep Learning MLP")

            st.metric(
                label="Predicted Risk Category",
                value=DISPLAY_NAMES[mlp_result]
            )

        with result_col2:

            st.markdown("#### Random Forest")

            st.metric(
                label="Predicted Risk Category",
                value=DISPLAY_NAMES[rf_result]
            )

        if mlp_result == rf_result:

            st.info(
                "Both models generated the same "
                "benchmark risk classification. "
                "Agreement does not establish clinical "
                "validity or diagnostic certainty."
            )

        else:

            st.info(
                "The models disagree on this record. "
                "This reflects differences in their "
                "learned classification patterns. "
                "Neither result should be treated "
                "as a clinical determination."
            )

        st.caption(
            "Predictions represent learned benchmark "
            "risk labels, not probabilities of actual "
            "maternal complications."
        )

        st.subheader("Input Measurements")

        input_table = pd.DataFrame(
            [{
                "Indicator": feature,
                "Value": value
            } for feature, value in input_values.items()]
        )

        st.dataframe(
            input_table,
            hide_index=True,
            use_container_width=True
        )


# ============================================================
# TAB 2 — MODEL PERFORMANCE
# ============================================================

with tab2:

    st.header("Independent Test Evaluation")

    st.write(
        "This section presents the final performance "
        "of the Deep Learning MLP and Random Forest "
        "on 202 held-out observations from the "
        "Maternal Health Risk benchmark dataset."
    )

    comparison_file = (
        EVALUATION_DIR / "final_model_comparison.csv"
    )

    if comparison_file.exists():

        comparison = pd.read_csv(comparison_file)

        required_columns = [
            "model",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "high_risk_sensitivity",
            "high_risk_false_negatives"
        ]

        comparison = comparison[required_columns]

        # ----------------------------------------------------
        # PERFORMANCE TABLE
        # ----------------------------------------------------

        st.subheader("Classification Performance")

        display = comparison.copy()

        for column in [
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "high_risk_sensitivity"
        ]:
            display[column] = (
                display[column] * 100
            ).map(lambda value: f"{value:.2f}%")

        display = display.rename(columns={
            "model": "Model",
            "accuracy": "Accuracy",
            "macro_f1": "Macro F1",
            "balanced_accuracy": "Balanced Accuracy",
            "high_risk_sensitivity": "High-Risk Sensitivity",
            "high_risk_false_negatives": "Missed High-Risk Cases"
        })

        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True
        )

        # ----------------------------------------------------
        # GROUPED BAR CHART — NOT STACKED
        # ----------------------------------------------------

        st.subheader("Model Performance Comparison")

        chart_data = comparison[[
            "model",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "high_risk_sensitivity"
        ]].melt(
            id_vars="model",
            var_name="Metric",
            value_name="Score"
        )

        chart_data["Score"] *= 100

        chart_data["Metric"] = chart_data[
            "Metric"
        ].replace({
            "accuracy": "Accuracy",
            "macro_f1": "Macro F1",
            "balanced_accuracy": "Balanced Accuracy",
            "high_risk_sensitivity": "High-Risk Sensitivity"
        })

        metric_order = [
            "Accuracy",
            "Macro F1",
            "Balanced Accuracy",
            "High-Risk Sensitivity"
        ]

        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Metric:N",
                    sort=metric_order,
                    title=None
                ),
                xOffset=alt.XOffset(
                    "model:N"
                ),
                y=alt.Y(
                    "Score:Q",
                    title="Performance (%)",
                    scale=alt.Scale(
                        domain=[0, 100]
                    )
                ),
                color=alt.Color(
                    "model:N",
                    title="Classification Model"
                ),
                tooltip=[
                    alt.Tooltip(
                        "model:N",
                        title="Model"
                    ),
                    alt.Tooltip(
                        "Metric:N"
                    ),
                    alt.Tooltip(
                        "Score:Q",
                        format=".2f",
                        title="Score (%)"
                    )
                ]
            )
            .properties(height=380)
        )

        st.altair_chart(
            chart,
            use_container_width=True
        )

        # ----------------------------------------------------
        # CONFUSION MATRICES
        # ----------------------------------------------------

        st.subheader("Confusion Matrices")

        cm1, cm2 = st.columns(2)

        with cm1:

            st.markdown("**Random Forest**")

            rf_cm = (
                EVALUATION_DIR /
                "random_forest_confusion_matrix.csv"
            )

            if rf_cm.exists():

                st.dataframe(
                    pd.read_csv(
                        rf_cm,
                        index_col=0
                    ),
                    use_container_width=True
                )

        with cm2:

            st.markdown("**Deep Learning MLP**")

            mlp_cm = (
                EVALUATION_DIR /
                "deep_learning_mlp_confusion_matrix.csv"
            )

            if mlp_cm.exists():

                st.dataframe(
                    pd.read_csv(
                        mlp_cm,
                        index_col=0
                    ),
                    use_container_width=True
                )

        st.caption(
            "Confusion matrix rows represent actual "
            "dataset labels; columns represent model "
            "predictions."
        )

        st.info(
            "Random Forest achieved higher overall "
            "accuracy on the held-out data, while the "
            "MLP identified slightly more high-risk-labelled "
            "observations. These differences do not "
            "establish clinical superiority."
        )

    else:

        st.warning(
            "Final evaluation results were not found."
        )


# ============================================================
# TAB 3 — ABOUT THE RESEARCH
# ============================================================

with tab3:

    st.header("About the Research Project")

    st.subheader("Research Title")

    st.write(
        "Maternal Complication Risk Classification "
        "and Detection Model for Zimbabwean Healthcare"
    )

    st.subheader("Research Background")

    st.write(
        "This MSc Data Science study investigates "
        "the use of deep learning for multiclass "
        "maternal risk classification using six "
        "physiological and demographic indicators."
    )

    st.write(
        "The study uses the publicly available "
        "Maternal Health Risk dataset and compares "
        "a Multilayer Perceptron neural network "
        "with traditional machine-learning algorithms."
    )

    st.subheader("Research Objectives")

    st.markdown(
        """
        1. To develop a deep learning model for
           maternal complication risk classification
           using routinely collected maternal
           health indicators.

        2. To evaluate and compare the performance
           of the deep learning model against
           selected traditional machine-learning
           algorithms.

        3. To identify and analyse the maternal
           health indicators that most influence
           model-based risk classification.
        """
    )

    st.subheader("Input Indicators")

    st.markdown(
        """
        1. Maternal age
        2. Systolic blood pressure
        3. Diastolic blood pressure
        4. Blood sugar
        5. Body temperature
        6. Heart rate
        """
    )

    st.subheader("Classification Categories")

    st.write(
        "The models classify observations into "
        "three categories defined by the source "
        "dataset: Low Risk, Medium Risk and High Risk."
    )

    st.subheader("Research Limitations")

    st.markdown(
        """
        - The dataset originates from maternal
          monitoring in Bangladesh, not Zimbabwe.
        - The source labels are not independently
          verified clinical complication diagnoses.
        - Some records contain identical physiological
          profiles with conflicting risk labels.
        - Only six physiological and demographic
          indicators are available.
        - The application has not undergone prospective
          clinical or external Zimbabwean validation.
        - The predictions must not be used to make
          clinical decisions.
        """
    )

    st.subheader("Research Ethics")

    st.info(
        "This application is intended for research, "
        "education and demonstrations using synthetic "
        "or public benchmark examples. Do not enter "
        "identifiable patient information."
    )

    st.divider()

    st.caption(
        "MaternalRisk AI | MSc Data Science "
        "Research Prototype | 2026"
    )

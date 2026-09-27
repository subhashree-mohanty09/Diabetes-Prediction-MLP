import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Diabetes Prediction — Academic ML Demonstration",
    page_icon="🩺",
    layout="centered"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.main-title {
    font-size: 30px;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 14px;
    color: #555;
    margin-bottom: 15px;
}

.info-box {
    background-color: #fff9d9;
    border-left: 5px solid #f0c419;
    padding: 12px 15px;
    border-radius: 4px;
    font-size: 13px;
    color: #555;
    margin-bottom: 15px;
}

.section-title {
    font-size: 21px;
    font-weight: 650;
    margin-top: 15px;
    margin-bottom: 3px;
}

.section-description {
    font-size: 13px;
    color: #777;
    margin-bottom: 12px;
}

label {
    font-size: 13px !important;
}

div.stButton > button {
    width: 100%;
    background-color: #ff2b2b;
    color: white;
    border: none;
    border-radius: 4px;
    height: 42px;
    font-weight: 600;
    font-size: 14px;
}

div.stButton > button:hover {
    background-color: #e60000;
    color: white;
}

.result-box {
    background-color: #ffe7e7;
    border-radius: 4px;
    padding: 12px;
    margin-top: 5px;
    font-size: 14px;
}

.probability-label {
    font-size: 12px;
    color: #777;
    margin-bottom: 0px;
}

.probability {
    font-size: 24px;
    font-weight: 700;
}

.divider {
    border-top: 1px solid #eeeeee;
    margin: 18px 0;
}

.model-info {
    font-size: 12px;
    color: #777;
    margin-top: 5px;
    margin-bottom: 15px;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">🩺 Diabetes Prediction — Academic<br>ML Demonstration</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine learning system for educational demonstration '
    'and academic evaluation of diabetes risk.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# INFORMATION BOX
# ==========================================================

st.markdown("""
<div class="info-box">
<b>Academic Use:</b> This application demonstrates how a machine learning
model can estimate diabetes risk from clinical attributes. It is intended
for academic and educational purposes only and should not be used as a
medical diagnosis.
</div>
""", unsafe_allow_html=True)


# ==========================================================
# LOAD MODEL AND PREPROCESSOR
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "diabetes_mlp.keras"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "diabetes_preprocessor.joblib"
)

model = load_model(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.markdown(
    '<div class="model-info">'
    '<b>Deployed Model:</b> Multi-Layer Perceptron (MLP)<br>'
    '<b>Dataset:</b> Pima Indians Diabetes Dataset<br>'
    '<b>ROC-AUC:</b> 0.8496'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# PATIENT ATTRIBUTES
# ==========================================================

st.markdown(
    '<div class="section-title">Patient Attributes</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Enter the patient information below.'
    '</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


# ==========================================================
# LEFT COLUMN
# ==========================================================

with col1:

    pregnancies = st.number_input(
        "Pregnancies (number of times pregnant)",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "Glucose (plasma glucose concentration)",
        min_value=0.0,
        max_value=250.0,
        value=120.0,
        step=1.0
    )

    blood_pressure = st.number_input(
        "Diastolic Blood Pressure (mm Hg)",
        min_value=0.0,
        max_value=150.0,
        value=70.0,
        step=1.0
    )

    skin_thickness = st.number_input(
        "Triceps Skin Fold Thickness (mm)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )


# ==========================================================
# RIGHT COLUMN
# ==========================================================

with col2:

    insulin = st.number_input(
        "2-Hour Serum Insulin (mu U/ml)",
        min_value=0.0,
        max_value=900.0,
        value=80.0,
        step=1.0
    )

    bmi = st.number_input(
        "Body Mass Index (BMI)",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.47,
        step=0.01
    )

    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=100,
        value=30,
        step=1
    )


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

if age < 30:
    age_group = "Young"
elif age < 50:
    age_group = "Middle"
else:
    age_group = "Senior"


if bmi < 18.5:
    bmi_category = "Underweight"
elif bmi < 25:
    bmi_category = "Normal"
elif bmi < 30:
    bmi_category = "Overweight"
else:
    bmi_category = "Obese"


if glucose < 100:
    glucose_level = "Normal"
elif glucose < 126:
    glucose_level = "Prediabetic"
else:
    glucose_level = "High"


glucose_bmi = glucose * bmi
glucose_age = glucose * age
bmi_age = bmi * age

insulin_log = np.log1p(insulin)


# ==========================================================
# INPUT DATAFRAME
# ==========================================================

input_data = pd.DataFrame({

    "Pregnancies": [pregnancies],

    "Glucose": [glucose],

    "BloodPressure": [blood_pressure],

    "SkinThickness": [skin_thickness],

    "Insulin": [insulin],

    "BMI": [bmi],

    "DiabetesPedigreeFunction": [diabetes_pedigree],

    "Age": [age],

    "AgeGroup": [age_group],

    "BMICategory": [bmi_category],

    "GlucoseLevel": [glucose_level],

    "Glucose_BMI": [glucose_bmi],

    "Glucose_Age": [glucose_age],

    "BMI_Age": [bmi_age],

    "Insulin_log": [insulin_log]
})


# ==========================================================
# PREDICT BUTTON
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "Predict Diabetes Risk"
)


# ==========================================================
# PREDICTION
# ==========================================================

if predict_button:

    # Preprocess input
    processed_input = preprocessor.transform(input_data)

    # Model prediction
    probability = model.predict(
        processed_input,
        verbose=0
    )[0][0]

    # Classification threshold used internally
    prediction = 1 if probability >= 0.5 else 0


    # ======================================================
    # RESULT SECTION
    # ======================================================

    st.markdown(
        "<div class='divider'></div>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Prediction Result</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns([1, 1])


    with result_col1:

        if prediction == 1:

            st.markdown(
                '<div class="result-box">'
                '<b>Predicted: Higher Risk</b>'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="result-box">'
                '<b>Predicted: Lower Risk</b>'
                '</div>',
                unsafe_allow_html=True
            )


    with result_col2:

        st.markdown(
            '<div class="probability-label">'
            'Estimated Probability of Diabetes'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="probability">{probability:.2%}</div>',
            unsafe_allow_html=True
        )


    # ======================================================
    # PROBABILITY BAR
    # ======================================================

    st.progress(
        float(probability)
    )


    # ======================================================
    # EXPLANATION
    # ======================================================

    st.markdown(
        """
        <div style="
            font-size:12px;
            color:#777;
            margin-top:8px;
            margin-bottom:15px;
        ">
        The probability shown above is generated by the trained MLP model
        based on the entered patient attributes. This output is intended
        for academic demonstration only and should not be interpreted
        as a clinical diagnosis.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# ABOUT ML APPLICATION
# ==========================================================

with st.expander("ℹ️ About ML Application"):

    st.write(
        """
        This application uses a Multi-Layer Perceptron (MLP) neural network
        for diabetes risk prediction.

        The model was trained using the Pima Indians Diabetes Dataset.

        Clinical attributes used by the model include:

        - Pregnancies
        - Glucose
        - Blood Pressure
        - Skin Thickness
        - Insulin
        - BMI
        - Diabetes Pedigree Function
        - Age

        Additional engineered features are generated before the data is
        passed through the trained preprocessing pipeline.

        Model evaluation:
        ROC-AUC = 0.8496

        The application is designed for academic demonstration and should
        not be interpreted as a clinical diagnostic tool.
        """
    )
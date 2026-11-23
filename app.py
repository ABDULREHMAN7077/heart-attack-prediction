import streamlit as st
import numpy as np
import joblib



st.set_page_config(
    page_title="Heart_attack Prediction App",
    page_icon="❤️",
    layout="centered"
)
model = joblib.load("linear_regression_model.pkl")
st.title("❤️ Heart Attack Prediction App")
st.write(
    "This machine learning application estimates the risk of a heart attack based on various health parameters.")
st.divider()
name = st.text_input("Enter The patients name")
st.subheader("👤 Patient Information")
age = st.number_input(
    "Age",
    min_value=0, 
    max_value=120, 
    value=5,
    step=5
)
gender = st.selectbox(
    "Gender",
    options=["Male", "Female"]
)
cp = st.selectbox(
    "Chest Pain Type",
    options=["Yes", "No"],
    help="Chest Pain Type: Yes = Typical Angina, No = Atypical Angina"
)
trestbps = st.number_input(
    "Resting Blood Pressure (in mm Hg)",
    min_value=0,
    max_value=300,
    value=120,
    step=10
)
chol = st.number_input(
    "Serum Cholesterol (in mg/dl)",
    min_value=0,
    max_value=1000,
    value=200,
    step=20
)
fbs = st.selectbox( 
    "Fasting Blood Sugar > 120 mg/dl",
    options=["Yes", "No"]
)
restecg = st.selectbox(
    "Resting Electrocardiographic Results",
    options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
)
thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=0,
    max_value=300,
    value=150,
    step=10
)   
exang = st.selectbox(
    "Exercise Induced Cardio",
    options=["Yes", "No"]
)
oldpeak = st.number_input(
    "ST Depression Induced by Exercise Relative to Rest",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.10
)
slope = st.selectbox(
    "Slope of the Peak Exercise ST Segment",
    options=["Upsloping", "Flat", "Downsloping"]
)
ca = st.number_input(
    "Number of Major Vessels (0-3) Colored by Fluoroscopy",
    min_value=0,
    max_value=3,
    value=0,
    step=1
)
thal = st.selectbox(
    "Thalassemia",
    options=["Normal", "Fixed Defect", "Reversible Defect"]
)
st.divider()
prediction_button = st.button("Predict Heart Attack Risk")  
if prediction_button:
    input_data = np.array([[age, 1 if gender == "Male" else 0,1 if  cp == "Yes" else 0,1 if trestbps >= 140 else 0, 1 if chol >= 240 else 0, 1 if fbs == "Yes" else 0,1 if restecg == "ST-T Wave Abnormality" else 0,1 if thalach >= 150 else 0, 1 if exang == "Yes" else 0,1 if oldpeak >= 1.0 else 0, 1 if slope == "Upsloping" else 0,1 if ca == 0 else 0, 1 if thal == "Normal" else 0]])
    risk_percentage = model.predict_proba(input_data)[0][1] * 100
    st.success(f"The predicted risk of a heart attack for {name} is: {risk_percentage:.2f}%")
    st.metric(label="Risk Percentage", value=f"{risk_percentage:.2f}%")
    st.balloons()
    st.info("Note: This prediction is based on a machine learning model and should not be considered a medical diagnosis. Please consult a healthcare professional for accurate assessment and advice.Otherwise we are not responsible for any health issues that may arise from the use of this application. ")
else:
    st.warning("Please enter the patient's information and click 'Predict Heart Attack Risk' to get the prediction.")   
st.divider()
with st.expander("ℹ️ About this App"):
    st.write(
        "This Heart Attack Prediction App is a machine learning application that estimates the risk of a heart attack based on various health parameters. "
        "The model used in this app has been trained on a dataset of patient information and their corresponding heart attack outcomes. "
        "Please note that this prediction is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition."
    )
    
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import csv
import os

# Load models and scaler
log_model = joblib.load("log_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Model accuracies
log_acc = 0.78
xgb_acc = 0.85
svm_acc = 0.83
rf_acc = 0.87

# Page config
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Custom CSS for styling inputs
st.markdown("""
    <style>
    div[data-baseweb="input"] input {
        background-color:#FF5733 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💉 Diabetes Prediction ")

# Sidebar for model selection
st.sidebar.header("🔍 Select Model")
model_choice = st.sidebar.radio("Choose a model", ("Logistic Regression", "XGBoost", "SVM", "Random Forest"))

# Input fields
st.subheader("📋 Enter Patient Information")

pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Save prediction history
history_file = "prediction_history.csv"

# Create CSV file with headers if it doesn't exist
if not os.path.exists(history_file):
    with open(history_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin",
            "BMI", "Diabetes Pedigree Function", "Age",
            "Prediction", "Confidence", "Model", "Accuracy"
        ])

# Predict button
if st.button("Predict"):
    # Create input DataFrame
    input_df = pd.DataFrame([{
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict and get confidence
    if model_choice == "Logistic Regression":
        prediction = log_model.predict(input_scaled)[0]
        confidence = log_model.predict_proba(input_scaled)[0][1]
        acc = log_acc
    elif model_choice == "XGBoost":
        prediction = xgb_model.predict(input_scaled)[0]
        confidence = xgb_model.predict_proba(input_scaled)[0][1]
        acc = xgb_acc
    elif model_choice == "SVM":
        prediction = svm_model.predict(input_scaled)[0]
        decision_score = svm_model.decision_function(input_scaled)[0]
        confidence = 1 / (1 + np.exp(-decision_score))  # Sigmoid normalization
        acc = svm_acc
    else:  # Random Forest
        prediction = rf_model.predict(input_scaled)[0]
        confidence = rf_model.predict_proba(input_scaled)[0][1]
        acc = rf_acc

    # Determine risk level
    if confidence >= 0.8:
        risk_level = "🔴 High Risk"
    elif confidence >= 0.5:
        risk_level = "🟠 Medium Risk"
    else:
        risk_level = "🟢 Low Risk"

    # Display results
    st.markdown("---")
    st.subheader("📊 Prediction Result")
    st.write(f"**Prediction:** {'🩸 Diabetic' if prediction == 1 else '✅ Non-Diabetic'}")
    st.write(f"**Confidence:** {confidence:.2f}")
    st.write(f"**Model Accuracy:** {acc:.2f}")
    st.write(f"⚠️ **Risk Level:** {risk_level} ({confidence:.2%})")

    # Save to CSV (only save the prediction data)
    row = [
        pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age,
        "Diabetic" if prediction == 1 else "Non-Diabetic",
        f"{confidence:.2%}", model_choice, acc
    ]
    with open(history_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)


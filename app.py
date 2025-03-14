import streamlit as st
import numpy as np
import joblib  # Load trained models

# Load saved models
def load_model(model_path):
    return joblib.load(model_path)

diabetes_model = load_model("diabetes.pkl")
heart_model = load_model("heart.pkl")
parkinsons_model = load_model("parkinsons.pkl")
liver_model = load_model("liver.pkl")
kidney_model = load_model("kidney.pkl")
breast_cancer_model = load_model("breast_cancer.pkl")

# Prediction function
def predict_disease(model, input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return "Positive" if prediction[0] == 1 else "Negative"

st.title("🩺 Multi-Disease Prediction App")
st.sidebar.title("Select a Disease")

menu = ["Diabetes", "Heart Disease", "Parkinson’s", "Liver Disease", "Kidney Disease", "Breast Cancer"]
choice = st.sidebar.radio("Choose a Disease", menu)

st.write("👋 Welcome! Enter your details to check your health status.")

# Diabetes Prediction
if choice == "Diabetes":
    st.subheader("Diabetes Prediction")
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin Level", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    age = st.number_input("Age", min_value=0, step=1)
    if st.button("Predict Diabetes"):
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        result = predict_disease(diabetes_model, input_data)
        st.write(f"🩸 **Diabetes Prediction:** {result}")

# Heart Disease Prediction
elif choice == "Heart Disease":
    st.subheader("Heart Disease Prediction")
    age = st.number_input("Age", min_value=0, step=1)
    sex = st.radio("Sex", [0, 1])  # 0: Female, 1: Male
    cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3)
    trestbps = st.number_input("Resting Blood Pressure", min_value=0)
    chol = st.number_input("Cholesterol Level", min_value=0)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (1: Yes, 0: No)", [0, 1])
    restecg = st.number_input("Resting ECG Result (0-2)", min_value=0, max_value=2)
    thalach = st.number_input("Max Heart Rate Achieved", min_value=0)
    exang = st.radio("Exercise-Induced Angina (1: Yes, 0: No)", [0, 1])
    oldpeak = st.number_input("ST Depression", min_value=0.0, format="%.2f")
    slope = st.number_input("Slope of ST Segment (0-2)", min_value=0, max_value=2)
    ca = st.number_input("Major Vessels Colored by Fluoroscopy (0-4)", min_value=0, max_value=4)
    thal = st.number_input("Thalassemia Type (0-3)", min_value=0, max_value=3)
    if st.button("Predict Heart Disease"):
        input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        result = predict_disease(heart_model, input_data)
        st.write(f"💖 **Heart Disease Prediction:** {result}")

# Parkinson’s Prediction
elif choice == "Parkinson’s":
    st.subheader("Parkinson’s Disease Prediction")
    features = [st.number_input(f"Feature {i+1}") for i in range(22)]  # Assuming 22 features
    if st.button("Predict Parkinson’s"):
        result = predict_disease(parkinsons_model, features)
        st.write(f"🧠 **Parkinson’s Prediction:** {result}")

# Liver Disease Prediction
elif choice == "Liver Disease":
    st.subheader("Liver Disease Prediction")
    features = [st.number_input(f"Feature {i+1}") for i in range(10)]  # Assuming 10 features
    if st.button("Predict Liver Disease"):
        result = predict_disease(liver_model, features)
        st.write(f"🧬 **Liver Disease Prediction:** {result}")

# Kidney Disease Prediction
elif choice == "Kidney Disease":
    st.subheader("Kidney Disease Prediction")
    features = [st.number_input(f"Feature {i+1}") for i in range(15)]  # Assuming 15 features
    if st.button("Predict Kidney Disease"):
        result = predict_disease(kidney_model, features)
        st.write(f"🩸 **Kidney Disease Prediction:** {result}")

# Breast Cancer Prediction
elif choice == "Breast Cancer":
    st.subheader("Breast Cancer Prediction")
    features = [st.number_input(f"Feature {i+1}") for i in range(30)]  # Assuming 30 features
    if st.button("Predict Breast Cancer"):
        result = predict_disease(breast_cancer_model, features)
        st.write(f"🎗 **Breast Cancer Prediction:** {result}")

st.write("👨‍⚕️ **Made by Kishore** 🚀")

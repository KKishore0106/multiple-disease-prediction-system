import streamlit as st
import pickle
import numpy as np

# Load saved models
def load_model(model_path):
    with open(model_path, "rb") as file:
        return pickle.load(file)

# Load models
diabetes_model = load_model("diabetes_model.pkl")
heart_model = load_model("heart_disease_model.pkl")

# Streamlit UI
st.title("🤖 Medical Chatbot for Disease Prediction")
st.sidebar.title("Choose a Disease")

menu = ["Diabetes", "Heart Disease"]
choice = st.sidebar.radio("Select Disease", menu)

st.write("👋 Hello! I’m your AI health assistant. Let's check your health.")

# Function to predict disease
def predict_disease(model, input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return "Positive" if prediction[0] == 1 else "Negative"

# Diabetes Prediction
if choice == "Diabetes":
    st.subheader("Diabetes Prediction Chatbot")
    
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin Level", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    age = st.number_input("Age", min_value=0, step=1)
    
    if st.button("Predict Diabetes"):
        diabetes_input = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        result = predict_disease(diabetes_model, diabetes_input)
        st.write(f"🩺 **Diabetes Prediction:** {result}")

# Heart Disease Prediction
if choice == "Heart Disease":
    st.subheader("Heart Disease Prediction Chatbot")
    
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
        heart_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        result = predict_disease(heart_model, heart_input)
        st.write(f"💖 **Heart Disease Prediction:** {result}")

st.write("Made by Kishore 🚀")


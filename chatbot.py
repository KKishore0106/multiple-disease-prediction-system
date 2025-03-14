import streamlit as st
import joblib
import numpy as np

# Load ML Models
diabetes_model = joblib.load("diabetes_model.sav")
heart_model = joblib.load("heart_model.sav")
parkinson_model = joblib.load("parkinson_model.sav")

# Streamlit UI
st.title("🤖 Medical Chatbot for Disease Prediction")
st.sidebar.title("Choose a Disease")
menu = ["Diabetes", "Heart Disease", "Parkinson’s"]
choice = st.sidebar.radio("Select Disease", menu)

# Function to predict
def predict_disease(model, input_data):
    input_data = np.array(input_data).reshape(1, -1)  # Reshape for model
    prediction = model.predict(input_data)
    return "Positive 🛑" if prediction[0] == 1 else "Negative ✅"

# Chatbot UI
st.write("👋 Hello! I’m your AI health assistant. Let's check your health.")

if choice == "Diabetes":
    st.subheader("Diabetes Prediction Chatbot")
    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose Level", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin Level", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0)

    if st.button("Predict Diabetes"):
        user_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        result = predict_disease(diabetes_model, user_data)
        st.write(f"🤖 **Prediction:** {result}")

elif choice == "Heart Disease":
    st.subheader("Heart Disease Prediction Chatbot")
    age = st.number_input("Age", min_value=0)
    sex = st.radio("Sex", [0, 1])  # 0: Female, 1: Male
    cp = st.number_input("Chest Pain Type", min_value=0)
    trestbps = st.number_input("Resting Blood Pressure", min_value=0)
    chol = st.number_input("Cholesterol Level", min_value=0)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.number_input("Resting ECG", min_value=0)
    thalach = st.number_input("Max Heart Rate Achieved", min_value=0)
    exang = st.radio("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", min_value=0.0)
    slope = st.number_input("Slope of Peak Exercise ST", min_value=0)
    ca = st.number_input("Number of Major Vessels", min_value=0)
    thal = st.number_input("Thalassemia", min_value=0)

    if st.button("Predict Heart Disease"):
        user_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        result = predict_disease(heart_model, user_data)
        st.write(f"🤖 **Prediction:** {result}")

elif choice == "Parkinson’s":
    st.subheader("Parkinson’s Disease Prediction Chatbot")
    fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0)
    fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0)
    flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0)
    jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0)
    jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0)
    rap = st.number_input("MDVP:RAP", min_value=0.0)
    ppq = st.number_input("MDVP:PPQ", min_value=0.0)
    ddp = st.number_input("Jitter:DDP", min_value=0.0)
    shimmer = st.number_input("MDVP:Shimmer", min_value=0.0)
    shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0)
    apq3 = st.number_input("Shimmer:APQ3", min_value=0.0)
    apq5 = st.number_input("Shimmer:APQ5", min_value=0.0)
    apq = st.number_input("MDVP:APQ", min_value=0.0)
    dda = st.number_input("Shimmer:DDA", min_value=0.0)
    nhr = st.number_input("NHR", min_value=0.0)
    hnr = st.number_input("HNR", min_value=0.0)
    rpde = st.number_input("RPDE", min_value=0.0)
    dfa = st.number_input("DFA", min_value=0.0)
    spread1 = st.number_input("Spread1", min_value=0.0)
    spread2 = st.number_input("Spread2", min_value=0.0)
    d2 = st.number_input("D2", min_value=0.0)
    ppe = st.number_input("PPE", min_value=0.0)

    if st.button("Predict Parkinson’s Disease"):
        user_data = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]
        result = predict_disease(parkinson_model, user_data)
        st.write(f"🤖 **Prediction:** {result}")

st.write("🚀 **Developed by Kishore**")

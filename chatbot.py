import streamlit as st
import pickle
import os
import requests

# Rename .sav files to .pkl (Run once)
model_files = ["diabetes_model.sav", "heart_disease_model.sav", "parkinsons_model.sav"]
for model_file in model_files:
    if os.path.exists(model_file):
        os.rename(model_file, model_file.replace(".sav", ".pkl"))

# Load models
with open("diabetes_model.pkl", "rb") as f:
    diabetes_model = pickle.load(f)

with open("heart_disease_model.pkl", "rb") as f:
    heart_disease_model = pickle.load(f)

with open("parkinsons_model.pkl", "rb") as f:
    parkinsons_model = pickle.load(f)

# Hugging Face API Setup (For AI Chatbot Responses)
HUGGINGFACE_API_KEY = "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"
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

def chatbot_response(user_input):
    prompt = f"You are a medical assistant. The user provides health details for disease prediction. Analyze and provide suggestions.\nUser: {user_input}\nAssistant:"
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json().get("generated_text", "Unable to process request.")
    else:
        return f"Error {response.status_code}: {response.json()}"

# Streamlit UI
st.title("🤖 Medical Chatbot for Disease Prediction")
st.sidebar.title("Choose a Disease")
menu = ["Diabetes", "Heart Disease", "Parkinson’s"]
choice = st.sidebar.radio("Select Disease", menu)

st.write("👋 Hello! I’m your AI health assistant. Let's check your health.")

if choice == "Diabetes":
    st.subheader("Diabetes Prediction Chatbot")
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0, step=1)

    if st.button("Predict Diabetes"):
        input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]]
        result = diabetes_model.predict(input_data)[0]
        st.write("🩺 **Prediction:**", "Diabetic" if result == 1 else "Not Diabetic")

elif choice == "Heart Disease":
    st.subheader("Heart Disease Prediction Chatbot")
    age = st.number_input("Age", min_value=0, step=1)
    sex = st.radio("Sex", [0, 1])
    cp = st.number_input("Chest Pain Type (cp)", min_value=0, max_value=3)
    trestbps = st.number_input("Resting Blood Pressure", min_value=0)
    chol = st.number_input("Cholesterol Level", min_value=0)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
    restecg = st.number_input("Resting ECG", min_value=0, max_value=2)
    thalach = st.number_input("Max Heart Rate Achieved", min_value=0)
    exang = st.radio("Exercise-Induced Angina (1 = Yes, 0 = No)", [0, 1])
    oldpeak = st.number_input("ST Depression", min_value=0.0)
    slope = st.number_input("Slope of Peak Exercise ST Segment", min_value=0, max_value=2)
    ca = st.number_input("Number of Major Vessels", min_value=0, max_value=4)
    thal = st.number_input("Thalassemia", min_value=0, max_value=3)

    if st.button("Predict Heart Disease"):
        input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        result = heart_disease_model.predict(input_data)[0]
        st.write("🩺 **Prediction:**", "Heart Disease Detected" if result == 1 else "No Heart Disease")

elif choice == "Parkinson’s":
    st.subheader("Parkinson’s Prediction Chatbot")
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

    if st.button("Predict Parkinson’s"):
        input_data = [[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]]
        result = parkinsons_model.predict(input_data)[0]
        st.write("🩺 **Prediction:**", "Parkinson’s Detected" if result == 1 else "No Parkinson’s")

st.write("Made by Kishore 🚀")

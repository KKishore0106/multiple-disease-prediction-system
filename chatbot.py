import streamlit as st
import pickle
import numpy as np
import requests

# Define API Key
HUGGINGFACE_API_KEY = "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"

# Hugging Face API Setup
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Function to get chatbot response
def chatbot_response(user_input):
    prompt = f"You are a medical assistant. The user provides health details for disease prediction. Analyze and provide suggestions.\nUser: {user_input}\nAssistant:"  
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json().get("generated_text", "Unable to process request.")
    else:
        return f"Error {response.status_code}: {response.json()}"

# Load Trained ML Models
diabetes_model = pickle.load(open("diabetes_model.pkl", "rb"))
heart_model = pickle.load(open("heart_model.pkl", "rb"))
parkinsons_model = pickle.load(open("parkinsons_model.pkl", "rb"))

# Streamlit UI
st.title("🤖 Medical Chatbot for Disease Prediction")
st.sidebar.title("Choose a Disease")
menu = ["Diabetes", "Heart Disease", "Parkinson’s"]
choice = st.sidebar.radio("Select Disease", menu)

# Function to predict disease
def predict_disease(model, user_inputs):
    user_inputs = np.array(user_inputs).reshape(1, -1)
    prediction = model.predict(user_inputs)[0]
    return "Positive (High Risk)" if prediction == 1 else "Negative (Low Risk)"

st.write("👋 Hello! I’m your AI health assistant. Let's check your health.")

if choice:
    st.subheader(f"{choice} Prediction Chatbot")
    
    # AI Chatbot for general health advice
    user_input = st.text_input("Enter your symptoms & details (e.g., age, glucose level, BMI)")
    if st.button("Chat with AI"):
        response = chatbot_response(user_input)
        st.write(f"🤖 **AI:** {response}")

    # Disease-specific prediction input
    if choice == "Diabetes":
        pregnancies = st.number_input("Pregnancies", min_value=0)
        glucose = st.number_input("Glucose Level", min_value=0)
        bp = st.number_input("Blood Pressure", min_value=0)
        skin = st.number_input("Skin Thickness", min_value=0)
        insulin = st.number_input("Insulin Level", min_value=0)
        bmi = st.number_input("BMI", min_value=0.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
        age = st.number_input("Age", min_value=0)

        if st.button("Predict Diabetes"):
            result = predict_disease(diabetes_model, [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age])
            st.write(f"🤖 **Prediction:** {result}")

    elif choice == "Heart Disease":
        age = st.number_input("Age", min_value=0)
        sex = st.selectbox("Sex", [0, 1])  # 0: Female, 1: Male
        cp = st.number_input("Chest Pain Type", min_value=0)
        trestbps = st.number_input("Resting Blood Pressure", min_value=0)
        chol = st.number_input("Cholesterol", min_value=0)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", min_value=0)
        exang = st.selectbox("Exercise-Induced Angina", [0, 1])
        oldpeak = st.number_input("ST Depression", min_value=0.0)
        slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
        ca = st.number_input("Number of Major Vessels", min_value=0)
        thal = st.selectbox("Thalassemia Type", [1, 2, 3])

        if st.button("Predict Heart Disease"):
            result = predict_disease(heart_model, [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
            st.write(f"🤖 **Prediction:** {result}")

    elif choice == "Parkinson’s":
        fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0)
        fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0)
        flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0)
        jitter = st.number_input("MDVP:Jitter(%)", min_value=0.0)
        shimmer = st.number_input("MDVP:Shimmer", min_value=0.0)
        hnr = st.number_input("HNR", min_value=0.0)
        rpde = st.number_input("RPDE", min_value=0.0)
        dfa = st.number_input("DFA", min_value=0.0)
        spread1 = st.number_input("Spread1", min_value=0.0)
        spread2 = st.number_input("Spread2", min_value=0.0)
        d2 = st.number_input("D2", min_value=0.0)
        ppe = st.number_input("PPE", min_value=0.0)

        if st.button("Predict Parkinson’s"):
            result = predict_disease(parkinsons_model, [fo, fhi, flo, jitter, shimmer, hnr, rpde, dfa, spread1, spread2, d2, ppe])
            st.write(f"🤖 **Prediction:** {result}")

st.write("Made by Kishore 🚀")

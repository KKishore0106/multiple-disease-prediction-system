import streamlit as st
import pickle
import numpy as np

# Load the models
def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

# Load all models
diabetes_model = load_model("diabetes_model.sav")
heart_model = load_model("heart_disease_model.sav")
parkinsons_model = load_model("parkinsons_model.sav")

# Streamlit UI
st.title("🤖 Medical Chatbot for Disease Prediction")
st.sidebar.title("Choose a Disease")
menu = ["Diabetes", "Heart Disease", "Parkinson’s"]
choice = st.sidebar.radio("Select Disease", menu)

st.write("👋 Hello! Enter your health details to get predictions.")

# User Input Form
def get_user_input(fields):
    user_data = []
    for field in fields:
        value = st.number_input(f"Enter {field}", min_value=0.0, format="%.2f")
        user_data.append(value)
    return np.array(user_data).reshape(1, -1)

if choice == "Diabetes":
    st.subheader("Diabetes Prediction")
    features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    user_input = get_user_input(features)
    if st.button("Predict"):
        prediction = diabetes_model.predict(user_input)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        st.write(f"🩺 **Prediction:** {result}")

elif choice == "Heart Disease":
    st.subheader("Heart Disease Prediction")
    features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    user_input = get_user_input(features)
    if st.button("Predict"):
        prediction = heart_model.predict(user_input)
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
        st.write(f"💖 **Prediction:** {result}")

elif choice == "Parkinson’s":
    st.subheader("Parkinson’s Prediction")
    features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", 
                "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", 
                "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]
    user_input = get_user_input(features)
    if st.button("Predict"):
        prediction = parkinsons_model.predict(user_input)
        result = "Parkinson’s Detected" if prediction[0] == 1 else "No Parkinson’s"
        st.write(f"🧠 **Prediction:** {result}")

st.write("🚀 Made by Kishore")

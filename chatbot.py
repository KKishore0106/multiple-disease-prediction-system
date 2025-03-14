import pickle
import streamlit as st

# Function to load model properly
def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

# Load models correctly
try:
    diabetes_model = load_model("diabetes_model.sav")
    heart_model = load_model("heart_disease_model.sav")
    parkinsons_model = load_model("parkinsons_model.sav")
except FileNotFoundError as e:
    st.error(f"🚨 Model file not found: {e}")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")

# Streamlit UI
st.title("🤖 Medical Chatbot for Disease Prediction")
st.sidebar.title("Choose a Disease")
menu = ["Diabetes", "Heart Disease", "Parkinson’s"]
choice = st.sidebar.radio("Select Disease", menu)

def get_user_input(choice):
    if choice == "Diabetes":
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        glucose = st.number_input("Glucose Level", min_value=0, step=1)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1)
        bmi = st.number_input("BMI", min_value=0.0, step=0.1)
        return [[age, glucose, blood_pressure, bmi]], diabetes_model

    elif choice == "Heart Disease":
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        cholesterol = st.number_input("Cholesterol Level", min_value=0, step=1)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1)
        thalach = st.number_input("Max Heart Rate", min_value=0, step=1)
        return [[age, cholesterol, blood_pressure, thalach]], heart_model

    elif choice == "Parkinson’s":
        fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, step=0.1)
        fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, step=0.1)
        flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, step=0.1)
        jitter = st.number_input("MDVP:Jitter(%)", min_value=0.0, step=0.01)
        return [[fo, fhi, flo, jitter]], parkinsons_model

    return None, None

if choice:
    st.subheader(f"{choice} Prediction")
    user_input, model = get_user_input(choice)

    if st.button("Predict"):
        if model:
            try:
                prediction = model.predict(user_input)[0]
                if prediction == 1:
                    st.error(f"⚠️ High risk of {choice}. Please consult a doctor.")
                else:
                    st.success(f"✅ No major signs of {choice}. Stay healthy!")
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.warning("Invalid input. Please enter valid health details.")

st.write("Made by Kishore 🚀")

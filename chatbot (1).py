import streamlit as st
import requests

# Use a smaller model
API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen1.5-7B-Chat"
API_KEY = "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"  # Replace this with st.secrets in deployment
headers = {"Authorization": f"Bearer {API_KEY}"}

# Function to get chatbot response
def chatbot_response(user_input):
    prompt = f"You are a medical assistant. The user provides health details for disease prediction. Analyze and provide suggestions.\nUser: {user_input}\nAssistant:"  
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json().get("generated_text", "Unable to process request.")
    else:
        return f"Error {response.status_code}: {response.text}"

# Streamlit UI
st.title("🤖 Medical Chatbot for Disease Prediction")
st.sidebar.title("Choose a Disease")
menu = ["Diabetes", "Heart Disease", "Parkinson’s", "Liver Disease", "Cancer"]
choice = st.sidebar.radio("Select Disease", menu)

# Chatbot-like UI
st.write("👋 Hello! I’m your AI health assistant. Let's check your health.")

if choice:
    st.subheader(f"{choice} Prediction Chatbot")
    user_input = st.text_input("Enter your symptoms & details (e.g., age, glucose level, BMI, etc.)")
    
    if st.button("Chat with AI"):
        response = chatbot_response(user_input)
        st.write(f"🧑‍💻 **You:** {user_input}")
        st.write(f"🤖 **AI:** {response}")

st.write("Made by Kishore 🚀")

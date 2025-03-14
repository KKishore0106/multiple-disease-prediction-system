import streamlit as st
import requests

# Hugging Face API Setup
API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen-14B-Chat"
st.secrets["HUGGINGFACE_API_KEY"] = "YOUR_NEW_HUGGINGFACE_API_KEY"
headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}"}

# Function to get chatbot response
def chatbot_response(user_input):
    prompt = f"You are a medical assistant. The user provides health details for disease prediction. Analyze and provide suggestions.\nUser: {user_input}\nAssistant:"  
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json().get("generated_text", "Unable to process request.")

# Streamlit UI
st.title("🤖 Medical Chatbot for Disease Prediction")
st.sidebar.title("Choose a Disease")
menu = ["Diabetes", "Heart Disease", "Parkinson’s", "Liver Disease", "Cancer"]
choice = st.sidebar.radio("Select Disease", menu)

# Chatbot-like UI
st.write("👋 Hello! I’m your AI health assistant. Let's check your health.")
chat_history = []

if choice == "Diabetes":
    st.subheader("Diabetes Prediction Chatbot")
    user_input = st.text_input("Enter your symptoms & details (e.g., age, glucose level, BMI)")
    if st.button("Chat with AI"):
        response = chatbot_response(user_input)
        chat_history.append((user_input, response))

elif choice == "Heart Disease":
    st.subheader("Heart Disease Prediction Chatbot")
    user_input = st.text_input("Enter your symptoms & details (e.g., age, cholesterol, blood pressure)")
    if st.button("Chat with AI"):
        response = chatbot_response(user_input)
        chat_history.append((user_input, response))

# Similar sections for Parkinson’s, Liver Disease, and Cancer can be added

# Display chat history
for user_msg, bot_reply in chat_history:
    st.write(f"🧑‍💻 **You:** {user_msg}")
    st.write(f"🤖 **AI:** {bot_reply}")

st.write("Made by Kishore 🚀")
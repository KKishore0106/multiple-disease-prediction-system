import streamlit as st
import numpy as np
import pickle
import requests
import re

# **1️⃣ Set up page configuration**
st.markdown("""
<style>
    /* Full page background */
    .stApp {
        background: white;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar  */
    .stSidebar {
        background: #f0f2f6;
        color: black;
        border-right:2px solid #444;
        padding: 20px;
    }

    /* Sidebar text */
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar p {
        color: black !important;
    }

    /* Title & Subtitle */
    .title-container {
        padding-top: 40px;
        text-align: center;
    }

    .main-title {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-top: 10px;
        color: #333;
    }

    /* Chat area */
    .block-container {
        background: white;
        border-radius: 20px;
        padding: 15px 80px;
        max-width: 1000px; /* Prevents over-expansion */
        margin: left;
    }

    /* Chat container */
    .chat-container {
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        gap: 10px;
        padding: 5px;
    }

    /* Chat messages */
    .chat-message {
        max-width: 100%;
        padding: 20px;
        border-radius: 20px;
        font-size: 16px;
        text-align: bottom;
        word-wrap: break-word;
        margin: 15px 0;
    }

    /* User messages */
    .chat-message.user {
        background-color: #dbeafe;
        color: #1e3a8a;
        border-left: 5px solid #3b82f6;
        align-self: flex-end;
    }

    /* AI responses */
    .chat-message.assistant {
        background-color: #f3f4f6;
        color: black;
        border-left: 5px solid #22c55e;
        align-self: flex-start;
    }

    /* Chat input box */
    .stTextInput>div {
    position: fixed;
    bottom: 20px; /* Adjust as needed */
    left: 50%;
    transform: translateX(-50%);
    width: 80%; /* Adjust width */
    max-width: 1500px;
    background-color: white;
    border-radius: 20px;
    padding: 10px;
    z-index: 9999;
    }

    /* Chat input hover and focus effects */
    .stTextInput>div>div>input:hover,
    .stTextInput>div>div>input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 5px rgba(59, 130, 246, 0.5);
        outline: none;
    }

    /* Floating send button */
    .stButton>button.send-button {
        background-color: #3b82f6 !important;
        color: white !important;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        position: absolute;
        right: 5px;
        bottom: 5px;
        transition: all 0.2s ease-in-out;
        display: flex;
        align-items: right;
        justify-content: center;
        font-size: 18px;
    }

    .stButton>button.send-button:hover {
        background-color: #2563eb !important;
        transform: scale(1.1);
    }
</style>

<div class="title-container">
    <h1 style="color: #333;">Medical AI Assistant</h1>
    <p style="color: #555; font-size: 16px;">Chat with our AI to check for diseases or get health advice</p>
</div>
""", unsafe_allow_html=True)


# **2️⃣ Load ML Models (Optimized with Cache)**
@st.cache_resource
def load_model(filename):
    """Loads and caches the ML model to prevent reloading."""
    try:
        return pickle.load(open(filename, 'rb'))
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None

diabetes_model = load_model('diabetes.pkl')
heart_disease_model = load_model('heart.pkl')
parkinsons_model = load_model('parkinsons_model.pkl')
liver_model = load_model('liver.pkl')
kidney_model = load_model('kidney.pkl')
breast_cancer_model = load_model('breast_cancer.pkl')

# **3️⃣ Hugging Face API Setup (Cached) - Updated to use Llama 3.3 70B**
HF_API_TOKEN = "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"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"  # Updated to Llama 3.3 70B
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Modified function to chat with Llama 3.3 70B
@st.cache_data
def chat_with_llama(prompt, response_type="medical", input_values=None, disease=None, prediction=None):
    """Calls Hugging Face API with contextual prompting based on response_type"""
    try:
        # Different system prompts based on what we need
        if response_type == "medical":
            system_prompt = "You are a helpful medical AI assistant. Provide accurate health information without making diagnoses. Be concise but thorough."
        
        elif response_type == "risk_assessment":
            # Include the specific input values and prediction in the prompt for risk assessment
            system_prompt = f"""You are a medical AI assistant specializing in risk assessment. 
            
A patient has been tested for {disease} with the following values:
{format_input_values(input_values, disease)}

The ML model prediction is: {prediction}

Based on these values, evaluate the overall risk level (Low, Moderate, High, or Very High). 
Then provide 3-5 personalized health recommendations. Be specific about which values are concerning.
Format your response with clear headers and bullet points. Keep it under 300 words."""
        
        else:
            system_prompt = "You are a helpful AI assistant."
        
        # Create a proper instruction prompt for Llama 3.3
        formatted_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
            headers=headers,
            json={"inputs": formatted_prompt, "parameters": {"max_new_tokens": 500, "temperature": 0.7}}
        )
        data = response.json()
        
        # Extract the response text
        if isinstance(data, list):
            text = data[0]['generated_text']
            # Extract just the assistant's response
            if "<|im_start|>assistant" in text:
                text = text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
            return text
        elif isinstance(data, dict) and 'generated_text' in data:
            text = data['generated_text']
            if "<|im_start|>assistant" in text:
                text = text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
            return text
        else:
            return "⚠️ AI response error. Please try again."
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"

# Helper function to format input values for the LLM
def format_input_values(input_values, disease):
    """Format input values with their descriptions and normal ranges for LLM context"""
    if not input_values or not disease:
        return ""
    
    formatted_text = ""
    for field, value in input_values.items():
        field_info = disease_fields[disease][field]
        unit = f" {field_info['unit']}" if field_info['unit'] else ""
        formatted_text += f"- {field}: {value}{unit} (Normal range: {field_info['range']}{unit})\n"
    
    return formatted_text

# **4️⃣ Disease Fields with Descriptions and Normal Ranges**
disease_fields = {
    "Diabetes": {
        "Pregnancy Count": {"description": "Number of times pregnant", "range": "0-20", "unit": "times"},
        "Glucose Level": {"description": "Plasma glucose concentration (2 hours in an oral glucose tolerance test)", "range": "70-180", "unit": "mg/dL"},
        "Blood Pressure": {"description": "Diastolic blood pressure", "range": "60-120", "unit": "mm Hg"},
        "Skin Thickness": {"description": "Triceps skin fold thickness", "range": "0-100", "unit": "mm"},
        "Insulin Level": {"description": "2-Hour serum insulin", "range": "0-846", "unit": "mu U/ml"},
        "BMI": {"description": "Body mass index", "range": "18.5-40", "unit": "weight in kg/(height in m)²"},
        "Diabetes Pedigree Function": {"description": "Diabetes pedigree function (hereditary factor)", "range": "0.078-2.42", "unit": ""},
        "Age": {"description": "Age", "range": "21-81", "unit": "years"}
    },
    "Heart Disease": {
        "Age": {"description": "Age", "range": "20-100", "unit": "years"},
        "Sex": {"description": "Sex (0 = female, 1 = male)", "range": "0-1", "unit": ""},
        "Chest Pain Type": {"description": "Chest pain type (0-3)", "range": "0-3", "unit": ""},
        "Resting Blood Pressure": {"description": "Resting blood pressure", "range": "90-200", "unit": "mm Hg"},
        "Serum Cholesterol": {"description": "Serum cholesterol", "range": "100-600", "unit": "mg/dl"},
        "Fasting Blood Sugar": {"description": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)", "range": "0-1", "unit": ""},
        "Resting ECG Result": {"description": "Resting electrocardiographic results (0-2)", "range": "0-2", "unit": ""},
        "Max Heart Rate": {"description": "Maximum heart rate achieved", "range": "60-220", "unit": "bpm"},
        "Exercise-Induced Angina": {"description": "Exercise induced angina (1 = yes, 0 = no)", "range": "0-1", "unit": ""},
        "ST Depression": {"description": "ST depression induced by exercise relative to rest", "range": "0-6.2", "unit": "mm"},
        "Slope of ST": {"description": "Slope of the peak exercise ST segment (0-2)", "range": "0-2", "unit": ""},
        "Major Vessels": {"description": "Number of major vessels colored by fluoroscopy (0-4)", "range": "0-4", "unit": ""},
        "Thalassemia": {"description": "Thalassemia (0-3)", "range": "0-3", "unit": ""}
    },
    "Parkinson's": {
        "MDVP:Fo(Hz)": {"description": "Average vocal fundamental frequency", "range": "80-260", "unit": "Hz"},
        "MDVP:Fhi(Hz)": {"description": "Maximum vocal fundamental frequency", "range": "100-600", "unit": "Hz"},
        "MDVP:Flo(Hz)": {"description": "Minimum vocal fundamental frequency", "range": "60-240", "unit": "Hz"},
        "MDVP:Jitter(%)": {"description": "Measure of variation in fundamental frequency", "range": "0-2", "unit": "%"},
        "MDVP:Jitter(Abs)": {"description": "Absolute measure of variation in fundamental frequency", "range": "0-0.0001", "unit": "ms"},
        "MDVP:RAP": {"description": "Relative amplitude perturbation", "range": "0-0.02", "unit": ""},
        "MDVP:PPQ": {"description": "Five-point period perturbation quotient", "range": "0-0.02", "unit": ""},
        "Jitter:DDP": {"description": "Average absolute difference of differences between cycles", "range": "0-0.03", "unit": ""},
        "MDVP:Shimmer": {"description": "Local shimmer", "range": "0-0.2", "unit": ""},
        "MDVP:Shimmer(dB)": {"description": "Local shimmer in decibels", "range": "0-2", "unit": "dB"}
    },
    "Liver Disease": {
        "Age": {"description": "Age of the patient", "range": "4-90", "unit": "years"},
        "Total Bilirubin": {"description": "Total bilirubin level", "range": "0.1-50", "unit": "mg/dL"},
        "Direct Bilirubin": {"description": "Direct bilirubin level", "range": "0.1-20", "unit": "mg/dL"},
        "Alkaline Phosphotase": {"description": "Alkaline phosphotase level", "range": "20-300", "unit": "IU/L"},
        "SGPT": {"description": "Serum glutamic pyruvic transaminase level", "range": "1-300", "unit": "IU/L"},
        "SGOT": {"description": "Serum glutamic oxaloacetic transaminase level", "range": "1-300", "unit": "IU/L"},
        "Total Proteins": {"description": "Total proteins level", "range": "5.5-9", "unit": "g/dL"},
        "Albumin": {"description": "Albumin level", "range": "2.5-5.5", "unit": "g/dL"},
        "Albumin/Globulin Ratio": {"description": "Ratio of albumin to globulin", "range": "0.3-2.5", "unit": ""}
    },
    "Kidney Disease": {
        "Age": {"description": "Age of the patient", "range": "2-90", "unit": "years"},
        "Blood Pressure": {"description": "Blood pressure", "range": "50-180", "unit": "mm Hg"},
        "Specific Gravity": {"description": "Specific gravity of urine", "range": "1.005-1.030", "unit": ""},
        "Albumin": {"description": "Albumin level", "range": "0-5", "unit": ""},
        "Sugar": {"description": "Sugar level", "range": "0-5", "unit": ""},
        "Red Blood Cells": {"description": "Red blood cells (0 = normal, 1 = abnormal)", "range": "0-1", "unit": ""},
        "Pus Cell": {"description": "Pus cell (0 = normal, 1 = abnormal)", "range": "0-1", "unit": ""},
        "Pus Cell Clumps": {"description": "Pus cell clumps (0 = not present, 1 = present)", "range": "0-1", "unit": ""},
        "Bacteria": {"description": "Bacteria (0 = not present, 1 = present)", "range": "0-1", "unit": ""},
        "Blood Glucose Random": {"description": "Random blood glucose level", "range": "70-490", "unit": "mg/dL"},
        "Blood Urea": {"description": "Blood urea level", "range": "1.5-100", "unit": "mg/dL"},
        "Serum Creatinine": {"description": "Serum creatinine level", "range": "0.4-15", "unit": "mg/dL"},
        "Sodium": {"description": "Sodium level", "range": "111-160", "unit": "mEq/L"},
        "Potassium": {"description": "Potassium level", "range": "2.5-7.5", "unit": "mEq/L"},
        "Hemoglobin": {"description": "Hemoglobin level", "range": "3.1-17.8", "unit": "g/dL"}
    },
    "Breast Cancer": {
        "Radius Mean": {"description": "Mean of distances from center to points on the perimeter", "range": "6.5-28", "unit": "mm"},
        "Texture Mean": {"description": "Standard deviation of gray-scale values", "range": "9.7-40", "unit": ""},
        "Perimeter Mean": {"description": "Mean size of the core tumor", "range": "43-190", "unit": "mm"},
        "Area Mean": {"description": "Mean area of the core tumor", "range": "140-2500", "unit": "sq. mm"},
        "Smoothness Mean": {"description": "Mean of local variation in radius lengths", "range": "0.05-0.16", "unit": ""},
        "Compactness Mean": {"description": "Mean of perimeter^2 / area - 1.0", "range": "0.02-0.35", "unit": ""},
        "Concavity Mean": {"description": "Mean of severity of concave portions of the contour", "range": "0-0.43", "unit": ""},
        "Concave Points Mean": {"description": "Mean number of concave portions of the contour", "range": "0-0.2", "unit": ""},
        "Symmetry Mean": {"description": "Mean symmetry", "range": "0.1-0.3", "unit": ""},
        "Fractal Dimension Mean": {"description": "Mean 'coastline approximation' - 1", "range": "0.05-0.1", "unit": ""}
    }
}

# Common symptoms for each disease
disease_symptoms = {
    "Diabetes": [
        "frequent urination", "excessive thirst", "unexplained weight loss", "extreme hunger", 
        "sudden vision changes", "tingling in hands or feet", "fatigue", "dry skin", 
        "slow-healing sores", "frequent infections"
    ],
    
    "Heart Disease": [
        "chest pain", "chest discomfort", "chest pressure", "shortness of breath", 
        "pain in the neck, jaw, throat, upper abdomen or back", "pain in arms", 
        "nausea", "fatigue", "lightheadedness", "cold sweat", "palpitations"
    ],
    
    "Parkinson's": [
        "tremor", "shaking", "rigid muscles", "impaired posture", "balance problems", 
        "loss of automatic movements", "speech changes", "writing changes", 
        "slowed movement", "bradykinesia"
    ],
    
    "Liver Disease": [
        "yellowing skin", "yellowing eyes", "jaundice", "abdominal pain", "swelling", 
        "dark urine", "pale stool", "chronic fatigue", "nausea", "vomiting", 
        "loss of appetite", "itchy skin"
    ],
    
    "Kidney Disease": [
        "decreased urine output", "fluid retention", "swelling", "shortness of breath", 
        "fatigue", "confusion", "nausea", "weakness", "irregular heartbeat", 
        "chest pain", "high blood pressure", "foamy urine"
    ],
    
    "Breast Cancer": [
        "breast lump", "change in breast size", "change in breast shape", 
        "breast skin dimpling", "breast pain", "nipple retraction", 
        "red or flaky skin", "nipple discharge", "swelling under arm"
    ]
}

# **5️⃣ Predict Function - Modified to return both prediction and raw result**
def get_prediction(disease, input_values):
    try:
        # Convert input values to a NumPy array
        input_data = np.array(list(map(float, input_values.values()))).reshape(1, -1)
        
        # Determine which model to use
        if disease == "Diabetes" and diabetes_model:
            prediction = diabetes_model.predict(input_data)[0]
        elif disease == "Heart Disease" and heart_disease_model: 
            prediction = heart_disease_model.predict(input_data)[0]
        elif disease == "Parkinson's" and parkinsons_model:
            prediction = parkinsons_model.predict(input_data)[0]
        elif disease == "Liver Disease" and liver_model:
            prediction = liver_model.predict(input_data)[0]
        elif disease == "Kidney Disease" and kidney_model:
            prediction = kidney_model.predict(input_data)[0]
        elif disease == "Breast Cancer" and breast_cancer_model:
            prediction = breast_cancer_model.predict(input_data)[0]
        else:
            return "⚠️ Model not available.", None
        
        # Return binary prediction (0 = negative, 1 = positive) and the result text
        result = "Positive" if prediction == 1 else "Negative"
        result_text = f"Based on your inputs, the prediction for {disease} is: **{result}**"
        
        return result_text, result
    
    except ValueError:
        return "⚠️ Invalid input detected. Please enter numeric values only.", None
    except IndexError:
        return "⚠️ Prediction format error. Check model output format.", None
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}", None

# **6️⃣ Symptom Analyzer Function**
def analyze_symptoms(user_input):
    user_input = user_input.lower()
    
    found_diseases = []
    for disease, symptoms in disease_symptoms.items():
        matched_symptoms = [symptom for symptom in symptoms if symptom in user_input]
        if matched_symptoms:
            found_diseases.append({
                "disease": disease,
                "symptoms": matched_symptoms,
                "count": len(matched_symptoms)
            })
    
    # Sort by number of symptoms matched
    found_diseases.sort(key=lambda x: x["count"], reverse=True)
    
    if found_diseases:
        top_disease = found_diseases[0]
        if top_disease["count"] >= 2:  # At least two symptoms needed for a tentative suggestion
            response = f"I noticed you mentioned {', '.join(top_disease['symptoms'])}. "
            response += f"These could be associated with {top_disease['disease']}. "
            response += f"Would you like to check for {top_disease['disease']}? (yes/no)"
            return response, top_disease["disease"]
        else:
            response = "I noticed you mentioned some health concerns. Would you like to check for a specific disease? (Diabetes, Heart Disease, Parkinson's, Liver Disease, Kidney Disease, or Breast Cancer)"
            return response, None
    else:
        return None, None

# After all inputs have been collected, display a summary and ask for confirmation
def handle_completed_inputs(disease, input_values):
    """Display a summary of all collected information and ask for confirmation"""
    response = "✅ I've collected all the necessary information for your " + disease + " prediction.\n\n"
    response += "Here's a summary of what you provided:\n\n"
    
    # Format each input value with proper labeling
    for field, value in input_values.items():
        field_info = disease_fields[disease][field]
        unit = f" {field_info['unit']}" if field_info['unit'] else ""
        response += f"- **{field}**: {value}{unit}\n"
    
    response += "\nAre all these details correct? (yes/no)"
    
    # Update conversation state to confirming data
    st.session_state.conversation_state = "confirming_data"
    return response

# Handle user confirmation of collected data
def handle_data_confirmation(prompt):
    """Process user's confirmation of the collected data"""
    if any(x in prompt.lower() for x in ["yes", "yeah", "sure", "okay", "ok", "yep", "y", "correct"]):
        st.session_state.conversation_state = "ready_for_prediction"
        return "Thank you for confirming. Shall we proceed with the prediction now? (yes/no)"
    else:
        # User indicates data is not correct
        fields_list = ", ".join(list(st.session_state.input_values.keys()))
        return f"Which value would you like to modify? Please specify one of these fields: {fields_list}"

# Handle user's decision to proceed with prediction - MODIFIED for enhanced Llama analysis
def handle_prediction_confirmation(prompt):
    """Process user's decision to proceed with prediction and get Llama analysis"""
    if any(x in prompt.lower() for x in ["yes", "yeah", "sure", "okay", "ok", "yep", "y", "proceed"]):
        # Generate the prediction
        disease = st.session_state.disease_name
        prediction_result, prediction_binary = get_prediction(disease, st.session_state.input_values)
        
        # Store the raw prediction for future reference
        st.session_state.last_prediction = prediction_binary
        
        # Add Llama 3.3 risk assessment based on the values and prediction
        risk_assessment = chat_with_llama(
            f"Provide a risk assessment for {disease} based on the provided values", 
            response_type="risk_assessment",
            input_values=st.session_state.input_values,
            disease=disease,
            prediction=prediction_binary
        )
        
        # Combine the model prediction with Llama's risk assessment
        response = f"{prediction_result}\n\n**Risk Assessment and Recommendations:**\n{risk_assessment}\n\nPlease note: This is not a medical diagnosis. Always consult with a healthcare professional for proper evaluation and treatment."
        
        # Reset to general conversation state
        st.session_state.conversation_state = "general"
        return response
    else:
        # User doesn't want to proceed with prediction
        st.session_state.conversation_state = "general"
        return "No problem. Is there anything else I can help you with regarding your health?"

# **7️⃣ Input Handlers for Different Conversation States**
def handle_general_state(prompt):
    """Handle user input when in general conversation state"""
    # Check for disease testing requests
    if any(x in prompt.lower() for x in ["check", "test", "assess", "diagnose"]) and any(disease.lower() in prompt.lower() for disease in disease_fields.keys()):
        for disease in disease_fields.keys():
            if disease.lower() in prompt.lower():
                st.session_state.disease_name = disease
                st.session_state.input_values = {}
                st.session_state.field_keys = list(disease_fields[disease].keys())
                st.session_state.current_field_index = 0
                current_field = st.session_state.field_keys[0]
                field_info = disease_fields[disease][current_field]
                
                response = f"I'll help you check for {disease}. I'll need to collect some medical information.\n\n"
                response += f"First, please enter your {current_field} ({field_info['description']}). "
                response += f"Typical range: {field_info['range']} {field_info['unit']}"
                
                st.session_state.conversation_state = "collecting_inputs"
                return response
    
    # Check for symptoms
    elif "symptom" in prompt.lower() or any(symptom in prompt.lower() for disease_symptoms_list in disease_symptoms.values() for symptom in disease_symptoms_list):
        symptom_response, suggested_disease = analyze_symptoms(prompt)
        if symptom_response:
            if suggested_disease:
                st.session_state.disease_name = suggested_disease
                st.session_state.conversation_state = "suggesting_disease"
            return symptom_response
        else:
            # If no clear symptoms found, use Llama
            return chat_with_llama(f"The user said: '{prompt}'. Respond as a medical AI assistant but avoid making specific diagnoses. Instead, focus on general health information and asking clarifying questions. If they described symptoms, acknowledge them but suggest consulting a healthcare provider for proper diagnosis.")
    
    def is_greeting(text):
         """Check if text contains a greeting"""
         greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy"]
         return any(greeting in text.lower() for greeting in greetings)

    # Main response function
    if is_greeting(prompt):
         return "Hello! 👋 How are you feeling today? I'm your AI medical assistant. I can help answer health questions, check for diabetes, heart disease, Parkinson's, liver disease, kidney disease, or breast cancer, or discuss symptoms you might be experiencing."

    # Ensure all responses remain medical
    return chat_with_llama(prompt, response_type="medical")

def handle_suggesting_disease_state(prompt):
    """Handle user input when suggesting a disease to check"""
    if any(x in prompt.lower() for x in ["yes", "yeah", "sure", "okay", "ok", "yep", "y"]):
        st.session_state.input_values = {}
        st.session_state.field_keys = list(disease_fields[st.session_state.disease_name].keys())
        st.session_state.current_field_index = 0
        current_field = st.session_state.field_keys[0]
        field_info = disease_fields[st.session_state.disease_name][current_field]
        
        response = f"Great! Let's check for {st.session_state.disease_name}. I'll need some medical information.\n\n"
        response += f"First, please enter your {current_field} ({field_info['description']}). "
        response += f"Typical range: {field_info['range']} {field_info['unit']}"
        
        st.session_state.conversation_state = "collecting_inputs"
        return response
    else:
        st.session_state.conversation_state = "general"
        return "No problem. Is there another disease you'd like to check (Diabetes, Heart Disease, Parkinson's, Liver Disease, Kidney Disease, or Breast Cancer), or do you have other health questions I can help with?"

def handle_modifying_field(prompt):
    """Handle user input when modifying a previously entered field"""
    disease = st.session_state.disease_name
    fields = disease_fields[disease]
    field = st.session_state.modifying_field
    field_info = fields[field]
    
    try:
        float_value = float(prompt)  # Validate the input is a number
        
        # Check if value is within expected range
        range_min, range_max = map(float, field_info['range'].split('-'))
        if float_value < range_min or float_value > range_max:
            warning = f"⚠️ Warning: The value {float_value} is outside the typical range ({field_info['range']}). Are you sure this is correct? (yes/no)"
            st.session_state.conversation_state = "confirming_out_of_range"
            st.session_state.temp_value = float_value
            return warning
        
        # Update the value
        st.session_state.input_values[field] = float_value
        st.session_state.conversation_state = "collecting_inputs"
        
        # Check if all inputs have been collected
        if len(st.session_state.input_values) == len(st.session_state.field_keys):
            return handle_completed_inputs(disease, st.session_state.input_values)
        else:
            # Continue collecting inputs
            return handle_collecting_inputs("continue")
    
    except ValueError:
        return f"Please enter a valid number for {field}. {field_info['description']} (Typical range: {field_info['range']} {field_info['unit']})"

def handle_confirming_out_of_range(prompt):
    """Handle user confirmation for out-of-range values"""
    if any(x in prompt.lower() for x in ["yes", "yeah", "sure", "okay", "ok", "yep", "y"]):
        # User confirms the out-of-range value is correct
        field = st.session_state.modifying_field
        st.session_state.input_values[field] = st.session_state.temp_value
        st.session_state.conversation_state = "collecting_inputs"
        
        # Check if all inputs have been collected
        if len(st.session_state.input_values) == len(st.session_state.field_keys):
            return handle_completed_inputs(st.session_state.disease_name, st.session_state.input_values)
        else:
            # Continue collecting inputs
            return handle_collecting_inputs("continue")
    else:
        # User wants to re-enter the value
        field = st.session_state.modifying_field
        field_info = disease_fields[st.session_state.disease_name][field]
        st.session_state.conversation_state = "modifying_field"
        return f"Please enter a new value for {field}. {field_info['description']} (Typical range: {field_info['range']} {field_info['unit']})"

def handle_collecting_inputs(prompt):
    """Handle user input when collecting disease test inputs"""
    try:
        if prompt != "continue":
            # Try to parse the input as a float
            input_value = float(prompt)
            
            # Check if the input is within valid range
            current_field = st.session_state.field_keys[st.session_state.current_field_index]
            current_field_info = disease_fields[st.session_state.disease_name][current_field]
            range_min, range_max = map(float, current_field_info['range'].split('-'))
            
            if input_value < range_min or input_value > range_max:
                warning = f"⚠️ Warning: The value {input_value} is outside the typical range ({current_field_info['range']}). Are you sure this is correct? (yes/no)"
                st.session_state.conversation_state = "confirming_out_of_range"
                st.session_state.temp_value = input_value
                st.session_state.modifying_field = current_field
                return warning
            
            # Store the input and move to the next field
            st.session_state.input_values[current_field] = input_value
            st.session_state.current_field_index += 1
        
        # Check if we've collected all inputs
        if st.session_state.current_field_index >= len(st.session_state.field_keys):
            return handle_completed_inputs(st.session_state.disease_name, st.session_state.input_values)
        
        # Ask for the next input
        current_field = st.session_state.field_keys[st.session_state.current_field_index]
        field_info = disease_fields[st.session_state.disease_name][current_field]
        response = f"Thank you. Now, please enter your {current_field} "
        response += f"({field_info['description']}). "
        response += f"Typical range: {field_info['range']} {field_info['unit']}"
        return response
        
    except ValueError:
        # User entered something that isn't a number
        current_field = st.session_state.field_keys[st.session_state.current_field_index]
        field_info = disease_fields[st.session_state.disease_name][current_field]
        return f"Please enter a valid number for {current_field}. {field_info['description']} (Typical range: {field_info['range']} {field_info['unit']})"
    except Exception as e:
        return f"An error occurred: {str(e)}. Please try again."

# **8️⃣ Chat History Setup and Management**
if 'messages' not in st.session_state:
    st.session_state.messages = []
    initial_message = "Hello! 👋 I'm your AI medical assistant. I can help answer health questions, check for diabetes, heart disease, Parkinson's, liver disease, kidney disease, or breast cancer, or discuss symptoms you might be experiencing."
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

if 'conversation_state' not in st.session_state:
    st.session_state.conversation_state = "general"

# Set up state variables for disease testing
if 'disease_name' not in st.session_state:
    st.session_state.disease_name = None

if 'input_values' not in st.session_state:
    st.session_state.input_values = {}

if 'field_keys' not in st.session_state:
    st.session_state.field_keys = []

if 'current_field_index' not in st.session_state:
    st.session_state.current_field_index = 0

if 'modifying_field' not in st.session_state:
    st.session_state.modifying_field = None

if 'temp_value' not in st.session_state:
    st.session_state.temp_value = None

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# **9️⃣ Chat Interface with Enhanced Processing Flow**
# Display chat messages
for message in st.session_state.messages:
    with st.container():
        st.markdown(f"""
        <div class="chat-container">
            <div class="chat-message {'user' if message['role'] == 'user' else 'assistant'}">
                {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# User input form
prompt = st.text_input("Type your message here...", key="user_input", placeholder="Ask about health or check for diseases...")

# Process messages
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process input based on conversation state
    if st.session_state.conversation_state == "general":
        response = handle_general_state(prompt)
    
    elif st.session_state.conversation_state == "suggesting_disease":
        response = handle_suggesting_disease_state(prompt)
    
    elif st.session_state.conversation_state == "collecting_inputs":
        response = handle_collecting_inputs(prompt)
    
    elif st.session_state.conversation_state == "confirming_data":
        response = handle_data_confirmation(prompt)
    
    elif st.session_state.conversation_state == "ready_for_prediction":
        response = handle_prediction_confirmation(prompt)
    
    elif st.session_state.conversation_state == "modifying_field":
        if prompt in st.session_state.input_values:
            # User has specified which field to modify
            st.session_state.modifying_field = prompt
            field_info = disease_fields[st.session_state.disease_name][prompt]
            response = f"Please enter a new value for {prompt}. {field_info['description']} (Typical range: {field_info['range']} {field_info['unit']})"
        else:
            # User is providing the value for the field to modify
            response = handle_modifying_field(prompt)
    
    elif st.session_state.conversation_state == "confirming_out_of_range":
        response = handle_confirming_out_of_range(prompt)
    
    else:
        # Default fallback
        response = chat_with_llama(prompt, response_type="medical")
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to update UI
    st.rerun()

# **🔟 UI Setup**

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create a container for chat history
chat_container = st.container()

# Display chat messages from history on app rerun
with chat_container:
    for message in st.session_state.messages:
        role_class = "user" if message["role"] == "user" else "assistant"
        st.markdown(f'<div class="chat-message {role_class}">{message["content"]}</div>', unsafe_allow_html=True)

# Accept user input
if prompt := st.chat_input("Ask me about your health or symptoms..."):
    # Display user message in chat message container
    with chat_container:
        st.markdown(f'<div class="chat-message user">{prompt}</div>', unsafe_allow_html=True)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        def process_user_input(prompt):
            """Main function to process user input based on conversation state"""
    # Process input based on conversation state
    if st.session_state.conversation_state == "general":
        response = handle_general_state(prompt)
    
    elif st.session_state.conversation_state == "suggesting_disease":
        response = handle_suggesting_disease_state(prompt)
    
    elif st.session_state.conversation_state == "collecting_inputs":
        response = handle_collecting_inputs(prompt)
    
    elif st.session_state.conversation_state == "confirming_data":
        response = handle_data_confirmation(prompt)
    
    elif st.session_state.conversation_state == "ready_for_prediction":
        response = handle_prediction_confirmation(prompt)
    
    elif st.session_state.conversation_state == "modifying_field":
        if prompt in st.session_state.input_values:
            # User has specified which field to modify
            st.session_state.modifying_field = prompt
            field_info = disease_fields[st.session_state.disease_name][prompt]
            response = f"Please enter a new value for {prompt}. {field_info['description']} (Typical range: {field_info['range']} {field_info['unit']})"
        else:
            # User is providing the value for the field to modify
            response = handle_modifying_field(prompt)
    
    elif st.session_state.conversation_state == "confirming_out_of_range":
        response = handle_confirming_out_of_range(prompt)
    
    else:
        # Default fallback
        response = chat_with_llama(prompt, response_type="medical")
    
    return response
        # Display assistant response in chat message container
        st.markdown(f'<div class="chat-message assistant">{response}</div>', unsafe_allow_html=True)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with information
with st.sidebar:
    # Medical AI Image
    st.image("https://www.svgrepo.com/show/13664/stethoscope.svg", width=100)

    # About AI Medical Assistant
    st.header("🤖 About AI Medical Assistant")
    st.markdown("""
    This AI-powered chatbot offers:
    
    - **Health Assessment**: Evaluate your risk for 6 different health conditions  
    - **Symptom Analysis**: Discuss your symptoms for possible insights  
    - **Health Advice**: Get general health recommendations based on your concerns  
    """)

    st.divider()

    # Available Disease Tests
    st.header("🩺 Available Disease Tests")

    diseases = {
        "Heart Disease": "❤️",
        "Diabetes": "🩸",
        "Parkinson’s": "🧠",
        "Liver Disease": "🫀",
        "Kidney Disease": "🚰",
        "Breast Cancer": "🎗️"
    }

    for disease, emoji in diseases.items():
        with st.expander(f"{emoji} {disease}", expanded=True):  # Expands by default so names are visible
            st.write("")  # Keeps the expander open

    st.divider()

    # Reset Conversation Button
    if st.button("🔄 Reset Conversation", help="Click to clear chat history"):
        st.session_state["messages"] = []  # Clear chat history
        st.rerun()  # Refresh Streamlit UI


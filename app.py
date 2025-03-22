import streamlit as st
import numpy as np
import pickle
import requests
import re

# **1️⃣ Set up page configuration**
st.set_page_config(page_title="Medical AI Chatbot", layout="wide")
st.markdown("""
<style>
    /* Apply a smooth gradient background */
    .stApp {
        background: linear-gradient(135deg, #dfe9f3, #ffffff);
        font-family: 'Arial', sans-serif;
    }

    /* Container for chat messages */
    .chat-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 20px; /* Adds space between messages */
        padding: 10px;
    }

    .chat-message {
        max-width: 70%;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
        font-size: 16px;
        line-height: 1.5;
        text-align: left;
        word-wrap: break-word;
        animation: fadeIn 0.5s ease-in-out;
        margin-top: 15px;  /* Adds space above each response */
        margin-bottom: 15px; /* Adds space below each response */
    }

    /* User message style */
    .chat-message.user {
        background-color: #e9f5ff;
        border-left: 5px solid #3b82f6;
        align-self: flex-end;
    }

    /* Assistant message style */
    .chat-message.assistant {
        background-color: #f0f9ff;
        border-left: 5px solid #22c55e;
        align-self: flex-start;
    }

    /* Hover effect */
    .chat-message:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    }

    /* Button hover effect */
    .stButton>button {
        transition: all 0.3s ease-in-out;
        border-radius: 8px;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        background-color: #3b82f6 !important;
        color: white !important;
    }

    /* Fade-in animation */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>
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

# **3️⃣ Hugging Face API Setup (Cached)**
HF_API_TOKEN = "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"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Modify the chat_with_mistral function to handle different types of responses
@st.cache_data
def chat_with_mistral(prompt, response_type="medical"):
    """Calls Hugging Face API with contextual prompting based on response_type"""
    try:
        # Add specific context based on the type of response needed
        if response_type == "medical":
            system_prompt = "You are a helpful medical AI assistant. Provide accurate health information without making diagnoses. Be concise but thorough."
        elif response_type == "casual":
            system_prompt = "You are a friendly conversational AI. Keep responses short, engaging, and natural. Ask thoughtful follow-up questions."
        elif response_type == "educational":
            system_prompt = "You are an educational AI that explains medical concepts in simple terms. Use analogies when helpful."
        else:
            system_prompt = "You are a helpful AI assistant."
        
        # Create a proper instruction prompt for Mistral
        formatted_prompt = f"""<s>[INST] {system_prompt}

User message: {prompt} [/INST]</s>"""
        
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
            headers=headers,
            json={"inputs": formatted_prompt, "parameters": {"max_new_tokens": 250}}
        )
        data = response.json()
        
        # Clean up the response to extract just the answer
        if isinstance(data, list):
            text = data[0]['generated_text']
            # Extract everything after the last [/INST] tag
            if "[/INST]" in text:
                text = text.split("[/INST]")[-1].strip()
            return text
        else:
            return "⚠️ AI response error."
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"
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

# **5️⃣ Predict Function**
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
            return "⚠️ Model not available."
        
        # Return binary prediction (0 = negative, 1 = positive)
        result = "Positive" if prediction == 1 else "Negative"
        result_text = f"Based on your inputs, the prediction for {disease} is: **{result}**"
        
        return result_text
    
    except ValueError:
        return "⚠️ Invalid input detected. Please enter numeric values only."
    except IndexError:
        # Add specific handling for the indexing error
        return "⚠️ Prediction format error. Check model output format."
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"

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
    """Display a summary of all collected inputs and ask for confirmation"""
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

# Handle user's decision to proceed with prediction
def handle_prediction_confirmation(prompt):
    """Process user's decision to proceed with prediction"""
    if any(x in prompt.lower() for x in ["yes", "yeah", "sure", "okay", "ok", "yep", "y", "proceed"]):
        # Generate the prediction
        disease = st.session_state.disease_name
        prediction_result = get_prediction(disease, st.session_state.input_values)
        
        # Add contextual information based on the prediction
        if "Positive" in prediction_result:
            medical_advice = chat_with_mistral(f"Provide concise, helpful general information and lifestyle advice for someone who has tested positive for {disease}. Do not claim to diagnose the person and remind them to consult a healthcare professional.")
            response = f"{prediction_result}\n\n{medical_advice}\n\nPlease note: This is not a medical diagnosis. Always consult with a healthcare professional for proper evaluation and treatment."
        else:
            response = f"{prediction_result}\n\nEven with a negative result, please consult with a healthcare professional for comprehensive evaluation. Would you like information about preventive measures for {disease}?"
        
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
            # If no clear symptoms found, use Mistral
            return chat_with_mistral(f"The user said: '{prompt}'. Respond as a medical AI assistant but avoid making specific diagnoses. Instead, focus on general health information and asking clarifying questions. If they described symptoms, acknowledge them but suggest consulting a healthcare provider for proper diagnosis.")
    
    # For greetings
    elif is_greeting(prompt):
        return "Hello! 👋 How are you feeling today? I'm your AI medical assistant. I can help answer health questions, check for diabetes, heart disease, Parkinson's, liver disease, kidney disease, or breast cancer, or discuss symptoms you might be experiencing."
    
    
    # Detect if it's a very short casual message
    if len(prompt.split()) < 3:
        return chat_with_mistral(prompt, response_type="casual")
    else:
        return chat_with_mistral(prompt, response_type="medical")

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
            return f"⚠️ The value you entered ({float_value}) is outside the typical range ({field_info['range']}). Are you sure this is correct? (yes/no)"
        else:
            st.session_state.input_values[field] = float_value
            st.session_state.modifying_field = None
            
            # Confirm the change and show current progress
            response = f"✅ Updated {field} to {float_value}.\n\nHere's what we have so far:\n"
            for f, v in st.session_state.input_values.items():
                response += f"- {f}: {v}\n"
            
            if st.session_state.current_field_index < len(st.session_state.field_keys):
                current_field = st.session_state.field_keys[st.session_state.current_field_index]
                field_info = fields[current_field]
                response += f"\nPlease enter your {current_field} ({field_info['description']}). "
                response += f"Typical range: {field_info['range']} {field_info['unit']}"
            else:
                response += "\nIs all this information correct? (yes/no)"
                st.session_state.conversation_state = "confirming_data"
            
            return response
    except ValueError:
        return f"⚠️ Please enter a valid number for {st.session_state.modifying_field}."

def handle_collecting_field_input(prompt):
    """Handle numeric input for the current field"""
    disease = st.session_state.disease_name
    fields = disease_fields[disease]
    current_field = st.session_state.field_keys[st.session_state.current_field_index]
    field_info = fields[current_field]
    
    try:
        float_value = float(prompt)  # Validate the input is a number
        
        # Check if value is within expected range
        range_min, range_max = map(float, field_info['range'].split('-'))
        if float_value < range_min or float_value > range_max:
            return f"⚠️ The value you entered ({float_value}) is outside the typical range ({field_info['range']}). Are you sure this is correct? (yes/no)"
        else:
            # Store the value and move to next field
            st.session_state.input_values[current_field] = float_value
            st.session_state.current_field_index += 1
            
            # Check if we have all fields or need more
            if st.session_state.current_field_index < len(st.session_state.field_keys):
                next_field = st.session_state.field_keys[st.session_state.current_field_index]
                field_info = fields[next_field]
                response = f"Great! Now, please enter your {next_field} ({field_info['description']}). "
                response += f"Typical range: {field_info['range']} {field_info['unit']}"
            else:
                # All fields collected - now use handle_completed_inputs for a cleaner summary
                return handle_completed_inputs(disease, st.session_state.input_values)
            
            return response
    except ValueError:
        return f"I didn't understand that. Please enter a numeric value for {current_field}."

def handle_specific_field(prompt):
    """Handle when a user specifies a field to modify"""
    disease = st.session_state.disease_name
    fields = disease_fields[disease]
    
    for field in st.session_state.field_keys:
        if field.lower() in prompt.lower():
            st.session_state.modifying_field = field
            field_info = fields[field]
            return f"Sure, let's update your {field} value. Please enter a new value for {field} ({field_info['description']}). Typical range: {field_info['range']} {field_info['unit']}"
    
    return "I couldn't identify which field you want to modify. Please specify one of: " + ", ".join(st.session_state.field_keys)

def handle_out_of_range_confirmation(prompt):
    """Handle user confirmation for out-of-range values"""
    disease = st.session_state.disease_name
    fields = disease_fields[disease]
    current_field = st.session_state.field_keys[st.session_state.current_field_index]
    
    if any(x in prompt.lower() for x in ["yes", "yeah", "sure", "okay", "ok", "yep", "y"]):
        # Try to get the last value from message history
        try:
            # Find the latest message that might contain a numeric value
            for i in range(len(st.session_state.messages) - 1, -1, -1):
                message = st.session_state.messages[i]["content"]
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", message)
                if numbers:
                    last_value = float(numbers[0])
                    st.session_state.input_values[current_field] = last_value
                    st.session_state.current_field_index += 1
                    break
            
            if st.session_state.current_field_index < len(st.session_state.field_keys):
                next_field = st.session_state.field_keys[st.session_state.current_field_index]
                field_info = fields[next_field]
                response = f"Noted. Now, please enter your {next_field} ({field_info['description']}). "
                response += f"Typical range: {field_info['range']} {field_info['unit']}"
            else:
                # All fields collected - use handle_completed_inputs
                return handle_completed_inputs(disease, st.session_state.input_values)
            
            return response
        except:
            return f"Let's try again. Please enter a numeric value for {current_field}."
    elif any(x in prompt.lower() for x in ["no", "nope", "n"]):
        field_info = fields[current_field]
        return f"Let's try again. Please enter a new value for {current_field} ({field_info['description']}). Typical range: {field_info['range']} {field_info['unit']}"
    else:
        return f"I didn't understand that. Please confirm with 'yes' or 'no'."

def is_greeting(text):
    """Check if text contains a greeting"""
    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy"]
    return any(greeting in text.lower() for greeting in greetings)

# **8️⃣ Main Processing Logic**
def process_user_input(prompt):
    """Main function to process user input based on conversation state"""
    if st.session_state.conversation_state == "general":
        return handle_general_state(prompt)
    elif st.session_state.conversation_state == "suggesting_disease":
        return handle_suggesting_disease_state(prompt)
    
    elif st.session_state.conversation_state == "collecting_inputs":
        if any(field.lower() in prompt.lower() for field in st.session_state.field_keys):
            # User wants to modify a specific field
            return handle_specific_field(prompt)
        elif st.session_state.modifying_field is not None:
            # User is entering a new value for a specific field
            return handle_modifying_field(prompt)
        else:
            # User is entering a value for the current field
            return handle_collecting_field_input(prompt)
    
    elif st.session_state.conversation_state == "confirming_data":
        return handle_data_confirmation(prompt)
    
    elif st.session_state.conversation_state == "ready_for_prediction":
        return handle_prediction_confirmation(prompt)
    
    # Catch any unexpected states
    else:
        st.session_state.conversation_state = "general"
        return "Let's start over. How can I help with your health concerns today?"

# **9️⃣ Initialize Session State for Chat History**
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = "general"

if "disease_name" not in st.session_state:
    st.session_state.disease_name = None

if "input_values" not in st.session_state:
    st.session_state.input_values = {}

if "field_keys" not in st.session_state:
    st.session_state.field_keys = []

if "current_field_index" not in st.session_state:
    st.session_state.current_field_index = 0

if "modifying_field" not in st.session_state:
    st.session_state.modifying_field = None

# **🔟 UI Setup**
# Page header
st.title("Medical AI Assistant")
st.write("Chat with our AI to check for diseases or get health advice")

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
        
        # Process user input and get response
        response = process_user_input(prompt)
        
        # Display assistant response in chat message container
        st.markdown(f'<div class="chat-message assistant">{response}</div>', unsafe_allow_html=True)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with information
with st.sidebar:
    # Medical AI Image
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/AI_healthcare_icon.png/800px-AI_healthcare_icon.png", use_container_width=True)

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


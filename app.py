import streamlit as st
import numpy as np
import pickle
import requests
import re

# **1️⃣ Set up page configuration**
st.set_page_config(page_title="Medical AI Chatbot", layout="wide")

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
parkinsons_model = load_model('parkinsons_model.sav')
liver_model = load_model('liver.pkl')
breast_cancer_model = load_model('breast_cancer.pkl')
kidney_model = load_model('kidney.pkl')

# **3️⃣ Hugging Face API Setup (Cached)**
HF_API_TOKEN = "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"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

@st.cache_data
def chat_with_mistral(prompt):
    """Calls Hugging Face API and caches responses."""
    try:
        # Create a proper instruction prompt for Mistral
        formatted_prompt = f"""<s>[INST] {prompt} [/INST]</s>"""
        
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
            headers=headers,
            json={"inputs": formatted_prompt}
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
    # New disease fields
    "Liver Disease": {
        "Age": {"description": "Age", "range": "4-90", "unit": "years"},
        "Gender": {"description": "Gender (1 = male, 0 = female)", "range": "0-1", "unit": ""},
        "Total Bilirubin": {"description": "Total bilirubin", "range": "0.1-50", "unit": "mg/dL"},
        "Direct Bilirubin": {"description": "Direct bilirubin", "range": "0.1-20", "unit": "mg/dL"},
        "Alkaline Phosphatase": {"description": "Alkaline phosphatase", "range": "20-2500", "unit": "IU/L"},
        "SGPT Alanine Aminotransferase": {"description": "Alanine aminotransferase", "range": "1-2000", "unit": "IU/L"},
        "SGOT Aspartate Aminotransferase": {"description": "Aspartate aminotransferase", "range": "1-2000", "unit": "IU/L"},
        "Total Proteins": {"description": "Total proteins", "range": "2-10", "unit": "g/dL"},
        "Albumin": {"description": "Albumin", "range": "1-6", "unit": "g/dL"},
        "Albumin and Globulin Ratio": {"description": "Albumin and globulin ratio", "range": "0.3-2.5", "unit": ""}
    },
    "Breast Cancer": {
        "Radius Mean": {"description": "Mean of distances from center to points on perimeter", "range": "6-30", "unit": "mm"},
        "Texture Mean": {"description": "Standard deviation of gray-scale values", "range": "10-40", "unit": ""},
        "Perimeter Mean": {"description": "Perimeter size of the mass", "range": "40-190", "unit": "mm"},
        "Area Mean": {"description": "Area of the mass", "range": "140-2500", "unit": "mm²"},
        "Smoothness Mean": {"description": "Local variation in radius lengths", "range": "0.05-0.16", "unit": ""},
        "Compactness Mean": {"description": "Perimeter² / area - 1.0", "range": "0.02-0.35", "unit": ""},
        "Concavity Mean": {"description": "Severity of concave portions of the contour", "range": "0-0.5", "unit": ""},
        "Concave Points Mean": {"description": "Number of concave portions of the contour", "range": "0-0.2", "unit": ""},
        "Symmetry Mean": {"description": "Symmetry of the mass", "range": "0.1-0.3", "unit": ""},
        "Fractal Dimension Mean": {"description": "Coastline approximation - 1", "range": "0.05-0.1", "unit": ""}
    },
    "Kidney Disease": {
        "Age": {"description": "Age", "range": "2-90", "unit": "years"},
        "Blood Pressure": {"description": "Blood pressure", "range": "50-180", "unit": "mm Hg"},
        "Specific Gravity": {"description": "Specific gravity of urine", "range": "1.005-1.025", "unit": ""},
        "Albumin": {"description": "Albumin level in urine", "range": "0-5", "unit": ""},
        "Sugar": {"description": "Sugar level in urine", "range": "0-5", "unit": ""},
        "Red Blood Cells": {"description": "Red blood cells in urine (1 = abnormal, 0 = normal)", "range": "0-1", "unit": ""},
        "Pus Cell": {"description": "Pus cell in urine (1 = abnormal, 0 = normal)", "range": "0-1", "unit": ""},
        "Pus Cell Clumps": {"description": "Pus cell clumps in urine (1 = present, 0 = not present)", "range": "0-1", "unit": ""},
        "Bacteria": {"description": "Bacteria in urine (1 = present, 0 = not present)", "range": "0-1", "unit": ""},
        "Blood Glucose Random": {"description": "Random blood glucose", "range": "70-500", "unit": "mg/dL"},
        "Blood Urea": {"description": "Blood urea", "range": "1.5-100", "unit": "mg/dL"},
        "Serum Creatinine": {"description": "Serum creatinine", "range": "0.4-15", "unit": "mg/dL"},
        "Sodium": {"description": "Sodium", "range": "110-150", "unit": "mEq/L"},
        "Potassium": {"description": "Potassium", "range": "2.5-7.5", "unit": "mEq/L"},
        "Hemoglobin": {"description": "Hemoglobin", "range": "3.1-17.8", "unit": "g/dL"}
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
    
    # New disease symptoms
    "Liver Disease": [
        "yellowing skin", "yellowing eyes", "jaundice", "abdominal pain", "swelling abdomen",
        "itchy skin", "dark urine", "pale stool", "fatigue", "nausea", "vomiting",
        "loss of appetite", "weight loss", "easy bruising", "swollen legs"
    ],
    
    "Breast Cancer": [
        "breast lump", "breast swelling", "breast pain", "nipple discharge", "nipple retraction", 
        "skin changes", "redness", "dimpling", "flaking skin", "breast asymmetry",
        "thickening", "swollen lymph nodes", "breast skin changes", "inverted nipple"
    ],
    
    "Kidney Disease": [
        "decreased urine output", "fluid retention", "swollen legs", "swollen ankles", 
        "shortness of breath", "fatigue", "confusion", "nausea", "weakness", "irregular heartbeat",
        "pressure in chest", "seizures", "high blood pressure", "metallic taste", "itchy skin", 
        "foamy urine", "back pain", "poor appetite"
    ]
}

# **5️⃣ Predict Function**
def get_prediction(disease, input_values):
    try:
        # Convert input values to a NumPy array
        input_data = np.array(list(map(float, input_values.values()))).reshape(1, -1)
        
        if disease == "Diabetes" and diabetes_model:
            # Check what predict returns
            if hasattr(diabetes_model, "predict_proba"):
                prediction = diabetes_model.predict_proba(input_data)[0][1]
            else:
                # If predict returns class labels directly
                raw_prediction = diabetes_model.predict(input_data)
                prediction = raw_prediction[0]
                
        elif disease == "Heart Disease" and heart_disease_model: 
            prediction = heart_disease_model.predict_proba(input_data)[0][1]
        
        elif disease == "Parkinson's" and parkinsons_model:
            prediction = parkinsons_model.predict_proba(input_data)[0][1]
        
        # New disease predictions
        elif disease == "Liver Disease" and liver_model:
            prediction = liver_model.predict_proba(input_data)[0][1]
            
        elif disease == "Breast Cancer" and breast_cancer_model:
            prediction = breast_cancer_model.predict_proba(input_data)[0][1]
            
        elif disease == "Kidney Disease" and kidney_model:
            prediction = kidney_model.predict_proba(input_data)[0][1]
        
        else:
            return "⚠️ Model not available.", None
        
        risk_level = "High" if prediction >= 0.7 else "Medium" if prediction >= 0.4 else "Low"
        prediction_percent = round(prediction * 100, 1)
        
        result_text = f"Based on your inputs, your risk level for {disease} is: **{risk_level}** ({prediction_percent}% probability)"
        return result_text, risk_level
    
    except ValueError:
        return "⚠️ Invalid input detected. Please enter numeric values only.", None
    except IndexError:
        # Add specific handling for the indexing error
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
            response += f"Would you like to check your risk for {top_disease['disease']}? (yes/no)"
            return response, top_disease["disease"]
        else:
            response = "I noticed you mentioned some health concerns. Would you like to check your risk for a specific disease? (Diabetes, Heart Disease, Parkinson's, Liver Disease, Breast Cancer, or Kidney Disease)"
            return response, None
    else:
        return None, None

# **7️⃣ Streamlit UI**
st.markdown("""
    <h1 style='text-align: center;'>🩺 AI Medical Chatbot</h1>
    <p style='text-align: center; font-size: 18px;'>Your AI assistant for health predictions and advice.</p>
""", unsafe_allow_html=True)

# **8️⃣ Initialize session state variables**
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial greeting
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "👋 Hello! I'm your AI medical assistant. I can help answer health questions, analyze symptoms, or assess your risk for diabetes, heart disease, Parkinson's disease, liver disease, breast cancer, or kidney disease. How can I help you today?"
    })

if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = "general"  # general, collecting_inputs, suggesting_disease
if "disease_name" not in st.session_state:
    st.session_state.disease_name = None
if "input_values" not in st.session_state:
    st.session_state.input_values = {}
if "current_field_index" not in st.session_state:
    st.session_state.current_field_index = 0
if "risk_level" not in st.session_state:
    st.session_state.risk_level = None
if "field_keys" not in st.session_state:
    st.session_state.field_keys = []
if "modifying_field" not in st.session_state:
    st.session_state.modifying_field = None
if "risk_assessed" not in st.session_state:  # Add this flag
    st.session_state.risk_assessed = False

# **9️⃣ Display chat history**
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍⚕️" if message["role"] == "assistant" else "🙂"):
        st.markdown(message["content"])

# **🔟 User input handling**
prompt = st.chat_input("Type your health question here...")

if prompt:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(prompt)
    
    # Initialize response
    response = ""
    
    # Check for greetings in general state
    greeting_patterns = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    is_greeting = any(greeting == prompt.lower().strip() for greeting in greeting_patterns)
    
    # GENERAL CONVERSATION STATE
    if st.session_state.conversation_state == "general":
        # Check for disease testing requests
        if any(x in prompt.lower() for x in ["check", "test", "assess", "diagnose"]) and any(disease.lower() in prompt.lower() for disease in disease_fields.keys()):
            for disease in disease_fields.keys():
                if disease.lower() in prompt.lower():
                    st.session_state.disease_name = disease
                    st.session_state.input_values = {}
                    st.session_state.field_keys = list(disease_fields[disease].keys())
                    st.session_state.current_field_index = 0
                    st.session_state.risk_assessed = False  # Reset risk assessment flag
                    current_field = st.session_state.field_keys[0]
                    field_info = disease_fields[disease][current_field]
                    
                    response = f"I'll help you assess your risk for {disease}. I'll need to collect some medical information.\n\n"
                    response += f"First, please enter your {current_field} ({field_info['description']}). "
                    response += f"Typical range: {field_info['range']} {field_info['unit']}"
                    
                    st.session_state.conversation_state = "collecting_inputs"
                    break
        # Check for symptoms
        elif "symptom" in prompt.lower() or any(symptom in prompt.lower() for disease_symptoms_list in disease_symptoms.values() for symptom in disease_symptoms_list):
            symptom_response, suggested_disease = analyze_symptoms(prompt)
            if symptom_response:
                response = symptom_response
                if suggested_disease:
                    st.session_state.disease_name = suggested_disease
                    st.session_state.conversation_state = "suggesting_disease"
            else:
                # If no clear symptoms found, use Mistral
                response = chat_with_mistral(f"The user said: '{prompt}'. Respond as a medical AI assistant but avoid making specific diagnoses. Instead, focus on general health information and asking clarifying questions. If they described symptoms, acknowledge them but suggest consulting a healthcare provider for proper diagnosis.")
        
        # For greetings or general questions, use Mistral
        elif is_greeting:
            response = "Hello! 👋 How are you feeling today? I'm your AI medical assistant. I can help answer health questions, check your risk for diabetes, heart disease, Parkinson's, liver disease, breast cancer, or kidney disease, or discuss symptoms you might be experiencing."
        else:
            # For general health questions
            response = chat_with_mistral(f"The user said: '{prompt}'. Respond as a medical AI assistant but avoid making specific diagnoses. Instead, focus on general health information and suggesting next steps. Always maintain a friendly and helpful tone.")
    
    # SUGGESTING DISEASE STATE
    elif st.session_state.conversation_state == "suggesting_disease":
        if any(x in prompt.lower() for x in ["yes", "yeah", "sure", "okay", "ok", "yep", "y"]):
            st.session_state.input_values = {}
            st.session_state.field_keys = list(disease_fields[st.session_state.disease_name].keys())
            st.session_state.current_field_index = 0
            st.session_state.risk_assessed = False  # Reset risk assessment flag
            current_field = st.session_state.field_keys[0]
            field_info = disease_fields[st.session_state.disease_name][current_field]
            
            response = f"Great! Let's check your risk for {st.session_state.disease_name}. I'll need some medical information.\n\n"
            response += f"First, please enter your {current_field} ({field_info['description']}). "
            response += f"Typical range: {field_info['range']} {field_info['unit']}"
            
            st.session_state.conversation_state = "collecting_inputs"
        else:
            response = "No problem. Is there another disease you'd like to check (Diabetes, Heart Disease, Parkinson's, Liver Disease, Breast Cancer, or Kidney Disease), or do you have other health questions I can help with?"
            st.session_state.conversation_state = "general"
    
    # COLLECTING INPUTS STATE
    elif st.session_state.conversation_state == "collecting_inputs":
        disease = st.session_state.disease_name
        fields = disease_fields[disease]
        
        # Make sure field_keys is properly initialized
        if not hasattr(st.session_state, 'field_keys') or st.session_state.field_keys is None:
            st.session_state.field_keys = list(fields.keys())
        
        # Make sure current_field_index is within bounds
        if st.session_state.current_field_index >= len(st.session_state.field_keys):
            st.session_state.current_field_index = len(st.session_state.field_keys) - 1
        
        # Now safely get the current field
        current_field = st.session_state.field_keys[st.session_state.current_field_index]
        
        # Handle risk assessment requests specially 
        if ("risk assessment" in prompt.lower() or any(x in prompt.lower() for x in ["get my results", "assess my risk"])) and len(st.session_state.input_values) > 0:
            # Only do assessment if we haven't already (prevents duplicate messages)
            if not st.session_state.risk_assessed:
                prediction_result, risk_level = get_prediction(disease, st.session_state.input_values)
                st.session_state.risk_level = risk_level
                st.session_state.risk_assessed = True  # Mark as assessed
                
                response = f"{prediction_result}\n\n"
                
                if risk_level == "High":
                    response += "⚠️ **Important:** This indicates a significant risk level. "
                    response += "I strongly recommend consulting with a healthcare professional for proper evaluation and diagnosis.\n\n"
                elif risk_level == "Medium":
                    response += "⚠️ This indicates a moderate risk level. "
                    response += "Consider discussing these results with a healthcare provider during your next visit.\n\n"
                else:  # Low
                    response += "✅ This indicates a low risk level. "
                    response += "Continue maintaining a healthy lifestyle and regular check-ups.\n\n"
                
                response += "Would you like some personalized health suggestions based on your risk level? (yes/no)"
                st.session_state.conversation_state = "post_assessment"
            else:
                # If already assessed, just ask about suggestions
                response = "Would you like some personalized health suggestions based on your risk level? (yes/no)"
                st.session_state.conversation_state = "post_assessment"
            
        # Check if user wants to modify a previous value
        elif any(pattern in prompt.lower() for pattern in ["change", "modify", "edit", "update", "fix", "correct", "redo"]) and st.session_state.current_field_index > 0:
            # Try to identify which field they want to change
            for field in st.session_state.field_keys:
                if field.lower() in prompt.lower() or any(word in prompt.lower() for word in field.lower().split()):
                    st.session_state.modifying_field = field
                    field_info = fields[field]
                    response = f"Sure, let's update your {field} value. Please enter a new value for {field} ({field_info['description']}). Typical range: {field_info['range']} {field_info['unit']}"
                    break
            
            if not response:  # If no specific field identified
                response = f"Which value would you like to change? Please specify one of: {', '.join(list(st.session_state.input_values.keys()))}"
        
        # If we're in modify mode, handle the new value
        elif st.session_state.modifying_field is not None:
            try:
                float_value = float(prompt)  # Validate the input is a number
                field = st.session_state.modifying_field
                field_info = fields[field]
                
                # Check if value is within expected range
                range_min, range_max = map(float, field_info['range'].split('-'))
                if float_value < range_min or float_value > range_max:
                    response = f"⚠️ The value you entered ({float_value}) is outside the typical range ({field_info['range']}). Are you sure this is correct? (yes/no)"
                else:
                    st.session_state.input_values[field] = float_value
                    st.session_state.modifying_field = None
                    st.session_state.risk_assessed = False  # Reset since data changed
                    
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
                        response += "\nWould you like to get your risk assessment now? (yes/no)"
            except ValueError:
                response = f"⚠️ Please enter a valid number for {st.session_state.modifying_field}."
        
        # Normal input collection flow
        else:
            try:
                float_value = float(prompt)  # Validate the input is a number
                
                current_field = st.session_state.field_keys[st.session_state.current_field_index]
                field_info = fields[current_field]
                
                # Check if value is within expected range
                range_min, range_max = map(float, field_info['range'].split('-'))
                if float_value < range_min or float_value > range_max:
                    response = f"⚠️ The value you entered ({float_value}) is outside the typical range ({field_info['range']}). Are you sure this is correct? (yes/no)"
                else:
                    # Store the value
                    st.session_state.input_values[current_field] = float_value
                    # Move to the next field
                    st.session_state.current_field_index += 1
                    
                    # If we have more fields to collect
                    if st.session_state.current_field_index < len(st.session_state.field_keys):
                        next_field = st.session_state.field_keys[st.session_state.current_field_index]
                        field_info = fields[next_field]
                        response = f"✅ Recorded your {current_field} as {float_value}.\n\n"
                        response += f"Now, please enter your {next_field} ({field_info['description']}). "
                        response += f"Typical range: {field_info['range']} {field_info['unit']}"
                    else:
                        # All inputs collected!
                        response = "✅ Thanks for providing all the necessary information. "
                        response += "Would you like to get your risk assessment now? (yes/no)"
            except ValueError:
                # If the input is not a number
                if "yes" in prompt.lower() and "range" in st.session_state.messages[-2]["content"]:
                    # They confirmed an out-of-range value
                    current_field = st.session_state.field_keys[st.session_state.current_field_index]
                    # Extract the number from their previous input (from chat history)
                    for msg in reversed(st.session_state.messages):
                        if msg["role"] == "user":
                            try:
                                # Try to find a number in the previous message
                                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", msg["content"])
                                if numbers and msg["content"] != prompt:
                                    float_value = float(numbers[0])
                                    st.session_state.input_values[current_field] = float_value
                                    st.session_state.current_field_index += 1
                                    
                                    if st.session_state.current_field_index < len(st.session_state.field_keys):
                                        next_field = st.session_state.field_keys[st.session_state.current_field_index]
                                        field_info = fields[next_field]
                                        response = f"✅ Recorded your {current_field} as {float_value}.\n\n"
                                        response += f"Now, please enter your {next_field} ({field_info['description']}). "
                                        response += f"Typical range: {field_info['range']} {field_info['unit']}"
                                    else:
                                        response = "✅ Thanks for providing all the necessary information. "
                                        response += "Would you like to get your risk assessment now? (yes/no)"
                                    break
                            except:
                                pass
                    
                    if not response:
                        response = f"I'm still looking for a valid number for {current_field}. Please enter just the number."
                else:
                    response = f"⚠️ Please enter a numeric value for {current_field}."
    
    # POST ASSESSMENT STATE
    elif st.session_state.conversation_state == "post_assessment":
        if any(x in prompt.lower() for x in ["yes", "yeah", "sure", "okay", "ok", "yep", "y"]):
            disease = st.session_state.disease_name
            risk_level = st.session_state.risk_level
            
            # Generate personalized health suggestions based on disease and risk level
            suggestion_prompt = f"""
            As a medical AI assistant, provide 3-5 personalized health suggestions for someone with {risk_level} risk of {disease}.
            Format as bullet points. Include lifestyle changes, monitoring advice, and when to see a doctor.
            Keep it concise, practical, and evidence-based. Don't include disclaimers or reminders that you're an AI.
            """
            
            # Get the AI response
            suggestions = chat_with_mistral(suggestion_prompt)
            
            response = f"Here are some personalized health suggestions based on your {risk_level.lower()} risk for {disease}:\n\n{suggestions}\n\n"
            response += "Remember that these are general suggestions and not a substitute for professional medical advice. Is there anything else I can help you with today?"
            
            st.session_state.conversation_state = "general"
        else:
            response = "No problem. Is there anything else I can help you with today? You can check your risk for other conditions or ask general health questions."
            st.session_state.conversation_state = "general"
    
    # Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant", avatar="🧑‍⚕️"):
        st.markdown(response)

# Add sidebar with app info and disclaimers
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=100)
    st.markdown("## About AI Medical Chatbot")
    st.markdown("This application helps assess your risk for various medical conditions based on your health metrics.")
    
    st.markdown("### Available Disease Assessments")
    for disease in disease_fields.keys():
        st.markdown(f"- {disease}")
    
    st.markdown("### How to Use")
    st.markdown("""
    1. Ask a health question or request risk assessment
    2. Enter your health metrics when prompted
    3. Review your risk assessment and suggestions
    """)
    
    st.markdown("### ⚠️ Medical Disclaimer")
    st.markdown("""
    This tool provides estimates based on statistical models and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for any medical concerns.
    """)
    
    st.markdown("### Privacy Notice")
    st.markdown("Your data is processed locally and not  stored or shared.")

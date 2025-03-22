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
parkinsons_model = load_model('parkinsons_model.pkl')

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
            response = "I noticed you mentioned some health concerns. Would you like to check your risk for a specific disease? (Diabetes, Heart Disease, or Parkinson's)"
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
        "content": "👋 Hello! I'm your AI medical assistant. I can help answer health questions, analyze symptoms, or assess your risk for diabetes, heart disease, or Parkinson's disease. How can I help you today?"
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
            response = "Hello! 👋 How are you feeling today? I'm your AI medical assistant. I can help answer health questions, check your risk for diabetes, heart disease, or Parkinson's, or discuss symptoms you might be experiencing."
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
            response = "No problem. Is there another disease you'd like to check (Diabetes, Heart Disease, or Parkinson's), or do you have other health questions I can help with?"
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
                        # We have all values, show summary and confirm
                        response = "Thanks for providing all the information. Here's a summary of what you entered:\n\n"
                        for field, value in st.session_state.input_values.items():
                            response += f"- {field}: {value}\n"
                        response += "\nWould you like to get your risk assessment now? (yes/no)"
            except ValueError:
                if any(x in prompt.lower() for x in ["yes", "yeah", "sure", "okay", "ok", "yep", "y"]):
                    # If they confirm an out-of-range value
                    current_field = st.session_state.field_keys[st.session_state.current_field_index]
                    try:
                        # Attempt to find the last numeric value from the chat history
                        last_value = None
                        for i in range(len(st.session_state.messages) - 2, 0, -1):
                            try:
                                last_value = float(st.session_state.messages[i]["content"])
                                break
                            except:
                                continue
                                
                        if last_value is not None:
                            st.session_state.input_values[current_field] = last_value
                            st.session_state.current_field_index += 1
                            
                            if st.session_state.current_field_index < len(st.session_state.field_keys):
                                next_field = st.session_state.field_keys[st.session_state.current_field_index]
                                field_info = fields[next_field]
                                response = f"Noted. Now, please enter your {next_field} ({field_info['description']}). "
                                response += f"Typical range: {field_info['range']} {field_info['unit']}"
                            else:
                                response = "Thanks for providing all the information. Here's a summary of what you entered:\n\n"
                                for field, value in st.session_state.input_values.items():
                                    response += f"- {field}: {value}\n"
                                response += "\nWould you like to get your risk assessment now? (yes/no)"
                        else:
                            response = f"Let's try again. Please enter a numeric value for {current_field}."
                    except:
                        response = f"Let's try again. Please enter a numeric value for {current_field}."
                elif any(x in prompt.lower() for x in ["no", "nope", "n"]):
                    # If they want to change an out-of-range value
                    current_field = st.session_state.field_keys[st.session_state.current_field_index]
                    field_info = fields[current_field]
                    response = f"Let's try again. Please enter a new value for {current_field} ({field_info['description']}). Typical range: {field_info['range']} {field_info['unit']}"
                else:
                    # Handle other non-numeric inputs during collection
                    current_field = st.session_state.field_keys[st.session_state.current_field_index]
                    response = f"⚠️ Please enter a valid number for {current_field}."
    
    # Special handling for final risk assessment request
    if st.session_state.conversation_state == "collecting_inputs" and len(st.session_state.input_values) == len(st.session_state.field_keys) and any(x in prompt.lower() for x in ["yes", "yeah", "sure", "okay", "ok", "yep", "y"]) and not st.session_state.risk_assessed:
        # Generate prediction
        prediction_result, risk_level = get_prediction(st.session_state.disease_name, st.session_state.input_values)
        st.session_state.risk_level = risk_level
        st.session_state.risk_assessed = True
        
        # Format response with prediction and next steps
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
    
    # POST ASSESSMENT STATE
    elif st.session_state.conversation_state == "post_assessment":
        if any(x in prompt.lower() for x in ["yes", "yeah", "sure", "okay", "ok", "yep", "y"]):
            # Generate personalized health suggestions
            disease = st.session_state.disease_name
            risk = st.session_state.risk_level
            
            suggestions_prompt = f"""
            As a medical AI assistant, provide personalized health suggestions for someone with {risk} risk of {disease}. 
            Include lifestyle modifications, diet recommendations, and when to seek medical attention. 
            Format the response in clear sections with bullet points. Keep it concise and practical.
            User values: {st.session_state.input_values}
            """
            
            response = chat_with_mistral(suggestions_prompt)
            response += "\n\nIs there anything specific you'd like to know about managing your health?"
        else:
            response = "Alright. Is there anything else I can help you with today?"
            st.session_state.conversation_state = "general"
    
    # Add assistant response to chat
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant", avatar="🧑‍⚕️"):
        st.markdown(response)

# Display sidebar with information
with st.sidebar:
    st.header("About This AI Medical Assistant")
    st.markdown("""
    This AI chatbot can help you:
    - Answer general health questions
    - Assess risk for diabetes, heart disease, and Parkinson's
    - Analyze symptoms you might
 - Provide personalized health suggestions
    
    **Important Disclaimer:** This tool provides general information only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions regarding a medical condition.
    """)

    st.header("How to Use")
    st.markdown("""
    1. **Ask health questions** - Type any health-related question
    2. **Check disease risk** - Say "Check my risk for diabetes" (or heart disease/Parkinson's)
    3. **Describe symptoms** - Tell the bot about any symptoms you're experiencing
    4. **Get suggestions** - Ask for personalized health recommendations
    
    You can always type "change [field name]" to modify a previously entered value.
    """)


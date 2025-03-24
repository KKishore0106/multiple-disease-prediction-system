import streamlit as st
import numpy as np
import pickle
import requests
import re
import json
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static

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
        align-self: flex-start;
    }

    /* AI responses */
    .chat-message.assistant {
        background-color: #f3f4f6;
        color: black;
        border-left: 5px solid #22c55e;
        align-self: flex-start;
    }

    /* Floating Chat Input Box */
        div[data-testid="stTextInput"] {
            position: fixed;
            bottom: 50px;
            left: 58%;
            transform: translateX(-50%);
            width: 80%;
            max-width: 800px;
            background-color: white;
            border-radius: 30px;
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

    /* Map container */
    .leaflet-container {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
</style>

<div class="title-container">
    <h1 style="color: #333;">Medical AI Assistant</h1>
    <p style="color: #555; font-size: 16px;">Chat with our AI to check for diseases, get health advice, and find healthcare facilities</p>
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

# **3️⃣ Hugging Face API Setup with Enhanced Intent Recognition**
HF_API_TOKEN = "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"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Define intent categories and their keywords
INTENT_CATEGORIES = {
    "greeting": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"],
    "medical_check": ["test", "check", "examine", "diagnose", "screen", "symptoms", "signs", "feeling", "pain", "discomfort", "not well"],
    "disease_inquiry": ["diabetes", "heart disease", "parkinsons", "liver", "kidney", "breast cancer", "cancer"],
    "hospital_search": ["hospital", "clinic", "doctor", "medical center", "healthcare", "facility", "where can i", "nearby"],
    "confirmation": ["yes", "yeah", "sure", "ok", "okay", "correct", "right", "confirm"],
    "denial": ["no", "nope", "not", "don't", "do not", "incorrect", "wrong"],
    "lifestyle": ["diet", "exercise", "nutrition", "sleep", "stress", "healthy", "habits", "routine"],
    "medication": ["medicine", "drug", "pill", "prescription", "treatment", "therapy"],
    "follow_up": ["next", "then", "what now", "after", "results", "outcome"],
    "farewell": ["bye", "goodbye", "see you", "thanks", "thank you"],
    "general_question": []  # Default category
}

@st.cache_data
def detect_intent(user_message):
    """Detects the user's intent from their message"""
    user_message = user_message.lower()
    
    # Check for disease name mentions first
    for disease in disease_symptoms.keys():
        if disease.lower() in user_message:
            return "disease_inquiry", disease
    
    # Check for symptom mentions
    symptoms_response, suggested_disease = analyze_symptoms(user_message)
    if symptoms_response and suggested_disease:
        return "medical_check", suggested_disease
    
    # Check for hospital search
    location_info = find_hospitals_from_input(user_message)
    if location_info[0]:  # If location was found
        return "hospital_search", location_info
    
        # Check for intent keywords
    for intent, keywords in INTENT_CATEGORIES.items():
        for keyword in keywords:
            if keyword in user_message:
                return intent, None
    
    # Default to general question if no specific intent is found
    return "general_question", None

@st.cache_data
def chat_with_mistral(prompt, response_type="medical", context=None):
    """Calls Hugging Face API with contextual prompting based on response_type"""
    try:
        # Detect intent first
        intent, intent_data = detect_intent(prompt)
        
        # Choose the appropriate system prompt based on intent
        if intent == "medical_check":
            system_prompt = """
            You are a helpful medical AI assistant. The user appears to be describing medical symptoms.
            Analyze these symptoms carefully, but do NOT provide a definitive diagnosis.
            Instead:
            1. Acknowledge the symptoms described
            2. Ask clarifying questions if needed
            3. Provide information about possible conditions associated with these symptoms
            4. ALWAYS emphasize the importance of consulting a healthcare professional
            5. If appropriate, suggest relevant tests that might be useful
            """
        elif intent == "disease_inquiry":
            system_prompt = f"""
            You are a helpful medical AI assistant. The user is asking about {intent_data if intent_data else 'a medical condition'}.
            Provide clear, factual information about this condition including:
            1. Brief overview of the condition
            2. Common symptoms
            3. Risk factors
            4. When to seek medical attention
            """
        elif intent == "hospital_search":
            system_prompt = """
            You are a helpful medical AI assistant. The user is looking for medical facilities.
            Help them by:
            1. Asking for their location if not provided
            2. Explaining what information you can provide about healthcare facilities
            3. Being clear about the importance of emergency services for urgent situations
            """
        elif intent == "lifestyle":
            system_prompt = """
            You are a helpful medical AI assistant. The user is asking about lifestyle recommendations.
            Provide evidence-based advice on:
            1. Diet and nutrition
            2. Physical activity
            3. Sleep hygiene
            4. Stress management
            Format your response in clear, actionable bullet points.
            """
        else:
            system_prompt = """
            You are a helpful medical AI assistant. Your role is to:
            1. Provide clear, accurate health information based on medical evidence
            2. Never diagnose conditions or prescribe treatments
            3. Always recommend consulting healthcare professionals for personal medical advice
            4. Be empathetic and supportive while maintaining appropriate medical boundaries
            5. Focus on evidence-based information and avoid speculation
            """
        
        # Add context if available
        if context:
            system_prompt += f"\n\nAdditional context: {context}"
            
        formatted_prompt = f"<s>[INST] {system_prompt}\n\nUser message: {prompt} [/INST]</s>"

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
            headers=headers,
            json={"inputs": formatted_prompt, "parameters": {"max_new_tokens": 500}}
        )

        # Check if request was successful
        if response.status_code != 200:
            return f"⚠️ API Error: {response.status_code} - {response.text}"

        data = response.json()

        # Ensure we have the expected response format
        if isinstance(data, list) and 'generated_text' in data[0]:
            text = data[0]['generated_text']
            if "[/INST]" in text:
                text = text.split("[/INST]")[-1].strip()
            return text
        else:
            return f"⚠️ Unexpected API Response: {data}"

    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"

# Function to get Mistral's interpretation of prediction results
def get_mistral_interpretation(disease, prediction_result, input_values):
    """Get Mistral's interpretation of the prediction results"""
    is_positive = "Positive" in prediction_result
    result_status = "positive" if is_positive else "negative"
    
    inputs_str = ", ".join([f"{k}: {v}" for k, v in input_values.items()])
    prompt = f"""
    Based on the following medical test values for {disease}, the prediction is {result_status}:
    {inputs_str}
    
    Please analyze these values and provide:
    1. An explanation of which values are concerning and why (if the prediction is positive)
    2. What these results might suggest about the patient's health
    3. Lifestyle or next steps recommendations based on these results
    4. Important monitoring or follow-up considerations
    
    Remember, this is not a diagnosis, but an analysis of test values.
    """
    
    interpretation = chat_with_mistral(prompt, "medical_analysis", f"Prediction: {result_status} for {disease}")
    return interpretation

# **3.1️⃣ Hospital Finder Functions**
def get_user_location(location_name):
    """Get latitude and longitude from a location name"""
    try:
        geolocator = Nominatim(user_agent="medical_assistant")
        location = geolocator.geocode(location_name)
        if location:
            return location.latitude, location.longitude
        return None
    except Exception as e:
        return None

def find_nearby_hospitals(latitude, longitude, radius=5000, disease_type=None):
    """Find nearby hospitals using simulated data with disease specialization"""
    
    try:
        # Simulated hospital data based on the given coordinates
        location_hash = hash(f"{latitude},{longitude}") % 1000
        
        # Base hospitals
        hospitals = [
            {
                "name": f"City General Hospital #{location_hash}",
                "vicinity": f"{int(location_hash/10)} Main Street",
                "rating": 4.5,
                "specialties": ["General Medicine", "Emergency Care", "Surgery"],
                "geometry": {"location": {"lat": latitude + 0.01, "lng": longitude + 0.01}}
            },
            {
                "name": f"Central Medical Center #{location_hash + 1}",
                "vicinity": f"{int(location_hash/5)} Oak Avenue",
                "rating": 4.2,
                "specialties": ["Cardiology", "Neurology", "Orthopedics"],
                "geometry": {"location": {"lat": latitude - 0.01, "lng": longitude - 0.01}}
            },
            {
                "name": f"St. Mary's Hospital #{location_hash + 2}",
                "vicinity": f"{int(location_hash/3)} Elm Street",
                "rating": 4.7,
                "specialties": ["Oncology", "Diabetes Care", "Renal Medicine"],
                "geometry": {"location": {"lat": latitude + 0.02, "lng": longitude - 0.02}}
            }
        ]
        
        # If a disease type is specified, add specialty hospitals and sort by relevance
        if disease_type:
            disease_map = {
                "Diabetes": ["Diabetes Care", "Endocrinology"],
                "Heart Disease": ["Cardiology", "Cardiovascular Surgery"],
                "Parkinson's": ["Neurology", "Movement Disorders"],
                "Liver Disease": ["Hepatology", "Gastroenterology"],
                "Kidney Disease": ["Nephrology", "Renal Medicine"],
                "Breast Cancer": ["Oncology", "Breast Cancer Center"]
            }
            
            specialties = disease_map.get(disease_type, [])
            
            # Add a specialty hospital for this disease
            if specialties:
                specialty_hospital = {
                    "name": f"{specialties[0]} Specialty Center #{location_hash + 3}",
                    "vicinity": f"{int(location_hash/2)} Specialist Boulevard",
                    "rating": 4.9,
                    "specialties": specialties,
                                        "geometry": {"location": {"lat": latitude - 0.015, "lng": longitude + 0.015}}
                }
                hospitals.append(specialty_hospital)
                
                # Sort hospitals by relevance to the disease
                def relevance_score(hospital):
                    hospital_specialties = hospital.get("specialties", [])
                    return sum(1 for s in specialties if s in hospital_specialties) * 2 + hospital.get("rating", 0)
                
                hospitals.sort(key=relevance_score, reverse=True)
        
        return hospitals
    except Exception as e:
        return []

def create_hospital_map(latitude, longitude, hospitals):
    """Create a map with hospital markers"""
    m = folium.Map(location=[latitude, longitude], zoom_start=13)
    
    # Add user location marker
    folium.Marker(
        [latitude, longitude],
        popup="Your Location",
        icon=folium.Icon(color="blue", icon="user", prefix="fa"),
    ).add_to(m)
    
    # Add hospital markers
    for hospital in hospitals:
        hospital_lat = hospital["geometry"]["location"]["lat"]
        hospital_lng = hospital["geometry"]["location"]["lng"]
        
        # Create popup with specialties if available
        specialties_html = ""
        if "specialties" in hospital:
            specialties_html = "<p><strong>Specialties:</strong> " + ", ".join(hospital["specialties"]) + "</p>"
            
        popup_html = f"""
        <div style="width:200px;">
            <h4>{hospital['name']}</h4>
            <p>{hospital['vicinity']}</p>
            <p>Rating: {hospital.get('rating', 'N/A')}/5</p>
            {specialties_html}
        </div>
        """
        
        folium.Marker(
            [hospital_lat, hospital_lng],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color="red", icon="plus", prefix="fa"),
        ).add_to(m)
    
    return m

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

# **5️⃣ Improved Prediction Function**
def get_prediction(disease, input_values):
    try:
        # Convert input values to a NumPy array
        input_data = np.array(list(map(float, input_values.values()))).reshape(1, -1)
        
        # Determine which model to use
        models = {
            "Diabetes": diabetes_model,
            "Heart Disease": heart_disease_model,
            "Parkinson's": parkinsons_model,
            "Liver Disease": liver_model,
            "Kidney Disease": kidney_model,
            "Breast Cancer": breast_cancer_model
        }
        
        model = models.get(disease)
        
        if not model:
            return "⚠️ Model not available for this disease."
        
        # Get prediction
        prediction = model.predict(input_data)[0]
        
        # Return binary prediction (0 = negative, 1 = positive)
        result = "Positive" if prediction == 1 else "Negative"
        result_text = f"Based on your inputs, the prediction for {disease} is: **{result}**"
        
        # Store result in session state for future reference
        st.session_state.prediction_result = result
        st.session_state.prediction_disease = disease
        st.session_state.prediction_inputs = input_values
        
        return result_text
    
    except ValueError:
        return "⚠️ Invalid input detected. Please enter numeric values only."
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"

# **6️⃣ Enhanced Symptom Analyzer**
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

# **6.1️⃣ Improved Hospital Finder Function**
def find_hospitals_from_input(user_input):
    """Extract location from user input and find hospitals"""
    disease_mentioned = None
    for disease in disease_symptoms.keys():
        if disease.lower() in user_input.lower():
            disease_mentioned = disease
            break
            
    location_keywords = ["near me", "nearby", "close to", "in", "around", "hospitals in", "doctors in", "clinics in"]
    has_location_keywords = any(keyword in user_input.lower() for keyword in location_keywords)
    
    location_patterns = [
        r"find (?:hospitals|doctors|clinics) (?:in|near|around) ([\w\s,]+)",
        r"hospitals (?:in|near|around) ([\w\s,]+)",
        r"medical (?:facilities|centers|care) (?:in|near|around) ([\w\s,]+)",
        r"(?:recommend|suggest) (?:hospitals|doctors|clinics) (?:in|near|around) ([\w\s,]+)"
    ]
    
    extracted_location = None
    
    if has_location_keywords:
        for pattern in location_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                extracted_location = match.group(1).strip()
                break
    
    if has_location_keywords and not extracted_location:
        for keyword in location_keywords:
            if keyword in user_input.lower():
                parts = user_input.lower().split(keyword)
                if len(parts) > 1 and parts[1].strip():
                    extracted_location = parts[1].strip()
                    extracted_location = re.sub(r'[?!.,;:]$', '', extracted_location).strip()
                    break
    
    if extracted_location:
        coordinates = get_user_location(extracted_location)
        if coordinates:
            latitude, longitude = coordinates
            hospitals = find_nearby_hospitals(latitude, longitude, disease_type=disease_mentioned)
            return extracted_location, coordinates, hospitals
    
    return None, None, None

# **7️⃣ Improved User Input Handling Functions**
def handle_completed_inputs(disease, input_values):
    """Display a summary of all collected information and ask for confirmation"""
    response = "✅ I've collected all the necessary information for your " + disease + " prediction.\n\n"
    response += "Here's a summary of what you provided:\n\n"
    
    for field, value in input_values.items():
        field_info = disease_fields[disease][field]
        unit = f" {field_info['unit']}" if field_info['unit'] else ""
        response += f"- **{field}**: {value}{unit}\n"
    
    response += "\nWould you like to proceed with the prediction? (yes/no)"
    return response

def handle_test_selection(disease):
    """Generate a response for the selected disease test"""
    fields = list(disease_fields[disease].keys())
    
    response = f"You've selected to check for {disease}. I'll need some medical information from you.\n\n"
    response += f"Please provide your {fields[0]}:"
    
    # Store the disease and remaining fields in session state
    st.session_state.current_disease = disease
    st.session_state.remaining_fields = fields[1:]
    st.session_state.collected_inputs = {}
    
    return response

def handle_field_input(user_input, disease, current_field):
    """Process user input for a specific medical field"""
    try:
        # Try to parse the input as a number
        value = float(user_input.strip())
        
        # Get the expected range for this field
        field_info = disease_fields[disease][current_field]
        range_str = field_info["range"]
        
        # Parse the range
        if "-" in range_str:
            min_val, max_val = map(float, range_str.split("-"))
            
            # Check if value is outside the normal range
            if value < min_val or value > max_val:
                response = f"⚠️ Your {current_field} value ({value}) is outside the normal range ({range_str}). "
                response += "Are you sure this value is correct? (yes/no)"
                st.session_state.confirming_abnormal_value = True
                st.session_state.temp_value = value
                return response
        
        # Store the value and move to next field
        st.session_state.collected_inputs[current_field] = value
        
        # Check if there are more fields
        if st.session_state.remaining_fields:
            next_field = st.session_state.remaining_fields[0]
            st.session_state.remaining_fields = st.session_state.remaining_fields[1:]
            response = f"Thanks! Now, please provide your {next_field}:"
            return response
        else:
            # All fields are collected
            st.session_state.awaiting_confirmation = True
            return handle_completed_inputs(disease, st.session_state.collected_inputs)
    
    except ValueError:
        return f"⚠️ Please enter a valid number for {current_field}."

def process_user_input(user_input):
    """Process user input based on current conversation state"""
    
    # Initialize session state variables if they don't exist
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'current_disease' not in st.session_state:
        st.session_state.current_disease = None
    if 'remaining_fields' not in st.session_state:
        st.session_state.remaining_fields = []
    if 'collected_inputs' not in st.session_state:
        st.session_state.collected_inputs = {}
    if 'confirming_abnormal_value' not in st.session_state:
        st.session_state.confirming_abnormal_value = False
    if 'temp_value' not in st.session_state:
        st.session_state.temp_value = None
    if 'awaiting_confirmation' not in st.session_state:
        st.session_state.awaiting_confirmation = False
    if 'suggested_disease' not in st.session_state:
        st.session_state.suggested_disease = None
    if 'hospital_search_location' not in st.session_state:
        st.session_state.hospital_search_location = None
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'prediction_disease' not in st.session_state:
        st.session_state.prediction_disease = None
    if 'prediction_inputs' not in st.session_state:
        st.session_state.prediction_inputs = {}
    if 'asking_for_interpretation' not in st.session_state:
        st.session_state.asking_for_interpretation = False
    
    # Add user message to conversation
    st.session_state.conversation.append({"role": "user", "content": user_input})
    
    # Handle different conversation states
    if st.session_state.confirming_abnormal_value:
        # User is confirming an abnormal value
        st.session_state.confirming_abnormal_value = False
        
        if any(word in user_input.lower() for word in ["yes", "yeah", "correct", "right", "sure", "ok", "okay"]):
            current_field = list(disease_fields[st.session_state.current_disease].keys())[len(st.session_state.collected_inputs)]
            st.session_state.collected_inputs[current_field] = st.session_state.temp_value
            
            # Check if there are more fields
            if st.session_state.remaining_fields:
                next_field = st.session_state.remaining_fields[0]
                st.session_state.remaining_fields = st.session_state.remaining_fields[1:]
                response = f"Noted. Now, please provide your {next_field}:"
            else:
                # All fields are collected
                st.session_state.awaiting_confirmation = True
                response = handle_completed_inputs(st.session_state.current_disease, st.session_state.collected_inputs)
        else:
            current_field = list(disease_fields[st.session_state.current_disease].keys())[len(st.session_state.collected_inputs)]
            response = f"Let's try again. Please provide a valid value for your {current_field}:"
    
    elif st.session_state.awaiting_confirmation:
        # User is confirming to proceed with prediction
        st.session_state.awaiting_confirmation = False
        
        if any(word in user_input.lower() for word in ["yes", "yeah", "correct", "right", "sure", "ok", "okay"]):
            # Get prediction from model
            prediction_result = get_prediction(st.session_state.current_disease, st.session_state.collected_inputs)
            
            # Get AI interpretation of the results
            if "Positive" in prediction_result:
                interpretation = get_mistral_interpretation(
                    st.session_state.current_disease, 
                    prediction_result, 
                    st.session_state.collected_inputs
                )
                response = f"{prediction_result}\n\n{interpretation}"
                response += "\n\n⚠️ **Important:** This is not a definitive diagnosis. Please consult with a healthcare professional for proper diagnosis and treatment."
                response += "\n\nWould you like me to help you find nearby healthcare facilities specialized in this condition?"
                st.session_state.asking_for_interpretation = False
            else:
                response = f"{prediction_result}\n\nThat's good news! Remember that this is just a screening tool. Regular check-ups with healthcare professionals are still important for your health."
                response += "\n\nWould you like me to provide more information about preventing this condition or any other health concerns?"
                st.session_state.asking_for_interpretation = True
        else:
            response = "I understand. The prediction has been canceled. Is there anything else I can help you with?"
            
            # Reset the current disease state
            st.session_state.current_disease = None
            st.session_state.remaining_fields = []
            st.session_state.collected_inputs = {}
    
    elif st.session_state.asking_for_interpretation:
        # User is asking for more information after prediction
        st.session_state.asking_for_interpretation = False
        
        if any(word in user_input.lower() for word in ["yes", "yeah", "sure", "ok", "okay", "please", "information", "tell me"]):
            if st.session_state.prediction_disease and st.session_state.prediction_result:
                # Get disease prevention or management information
                if st.session_state.prediction_result == "Negative":
                    prompt = f"Tell me how to prevent {st.session_state.prediction_disease} and maintain good health"
                else:
                    prompt = f"Tell me how to manage {st.session_state.prediction_disease} after a positive screening result"
                
                response = chat_with_mistral(prompt, "disease_management")
            else:
                response = "I'd be happy to provide general health information. What specific health topic would you like to know more about?"
        else:
            response = "No problem. Is there anything else I can help you with today?"
    
    elif st.session_state.current_disease and st.session_state.remaining_fields:
        # User is providing field values for disease prediction
        current_field_index = len(disease_fields[st.session_state.current_disease].keys()) - len(st.session_state.remaining_fields) - 1
        current_field = list(disease_fields[st.session_state.current_disease].keys())[current_field_index]
        response = handle_field_input(user_input, st.session_state.current_disease, current_field)
        
        if "Would you like to proceed with the prediction?" in response:
            st.session_state.awaiting_confirmation = True
    
    elif st.session_state.suggested_disease:
        # User is responding to a disease suggestion based on symptoms
        if any(word in user_input.lower() for word in ["yes", "yeah", "correct", "right", "sure", "ok", "okay"]):
            response = handle_test_selection(st.session_state.suggested_disease)
        else:
            response = "I understand. What type of health information or service are you looking for today?"
        
        st.session_state.suggested_disease = None
    
    elif st.session_state.hospital_search_location:
        # User provided or confirmed a location for hospital search
        location, coordinates, hospitals = st.session_state.hospital_search_location
        
        if coordinates and hospitals:
            disease_type = None
            
            # Check if searching for a disease-specific facility
            if "Positive" in str(st.session_state.prediction_result) and st.session_state.prediction_disease:
                disease_type = st.session_state.prediction_disease
                response = f"I found these healthcare facilities near {location}, specializing in {disease_type}:\n\n"
            else:
                response = f"I found these healthcare facilities near {location}:\n\n"
                
            for i, hospital in enumerate(hospitals, 1):
                response += f"{i}. **{hospital['name']}**\n"
                response += f"   Address: {hospital['vicinity']}\n"
                response += f"   Rating: {hospital.get('rating', 'N/A')}/5\n"
                if "specialties" in hospital:
                    response += f"   Specialties: {', '.join(hospital['specialties'])}\n"
                response += "\n"
            
            # Create and display the map
            latitude, longitude = coordinates
            map_obj = create_hospital_map(latitude, longitude, hospitals)
            st.session_state.display_map = (map_obj, location)
            
            response += "The map shows these locations relative to the address you provided."
        else:
            response = "I couldn't find any healthcare facilities for that location. Could you please provide a more specific location or city name?"
        
        st.session_state.hospital_search_location = None
    
    else:
        # New user query - detect intent and respond accordingly
        intent, intent_data = detect_intent(user_input)
        
        # Check if query is about finding hospitals for specific disease
        hospital_disease_match = False
        for disease in disease_symptoms.keys():
            if disease.lower() in user_input.lower() and any(term in user_input.lower() for term in ["hospital", "clinic", "facility", "doctor", "center"]):
                hospital_disease_match = True
                location_info = find_hospitals_from_input(user_input)
                
                if location_info[0]:  # If location was found
                    st.session_state.hospital_search_location = location_info
                    location, coordinates, hospitals = location_info
                    
                    if coordinates and hospitals:
                        response = f"I found these healthcare facilities near {location} that specialize in {disease}:\n\n"
                        
                        for i, hospital in enumerate(hospitals, 1):
                            response += f"{i}. **{hospital['name']}**\n"
                            response += f"   Address: {hospital['vicinity']}\n"
                            response += f"   Rating: {hospital.get('rating', 'N/A')}/5\n"
                            if "specialties" in hospital:
                                response += f"   Specialties: {', '.join(hospital['specialties'])}\n"
                            response += "\n"
                        
                        # Create and display the map
                        latitude, longitude = coordinates
                        map_obj = create_hospital_map(latitude, longitude, hospitals)
                        st.session_state.display_map = (map_obj, location)
                        
                        response += "The map shows these locations relative to the address you provided."
                    else:
                        response = "I couldn't find any healthcare facilities for that location. Could you please provide a more specific location or city name?"
                    
                    st.session_state.hospital_search_location = None
                else:
                    response = f"I can help you find hospitals specializing in {disease}. Please provide your city or location."
                break
        
        # If not a hospital search with disease, process based on intent
        if not hospital_disease_match:
            if intent == "disease_inquiry" and intent_data:
                response = handle_test_selection(intent_data)
            
            elif intent == "medical_check" and intent_data:
                response = handle_test_selection(intent_data)
                
            elif intent == "hospital_search" and intent_data[0]:
                location, coordinates, hospitals = intent_data
                
                if coordinates and hospitals:
                    response = f"I found these healthcare facilities near {location}:\n\n"
                    
                    for i, hospital in enumerate(hospitals, 1):
                        response += f"{i}. **{hospital['name']}**\n"
                        response += f"   Address: {hospital['vicinity']}\n"
                        response += f"   Rating: {hospital.get('rating', 'N/A')}/5\n"
                        if "specialties" in hospital:
                            response += f"   Specialties: {', '.join(hospital['specialties'])}\n"
                        response += "\n"
                    
                    # Create and display the map
                    latitude, longitude = coordinates
                    map_obj = create_hospital_map(latitude, longitude, hospitals)
                    st.session_state.display_map = (map_obj, location)
                    
                    response += "The map shows these locations relative to the address you provided."
                else:
                    response = "I couldn't find any healthcare facilities for that location. Could you please provide a more specific location or city name?"
            
            else:
                # Check for symptoms in the user input
                symptoms_response, suggested_disease = analyze_symptoms(user_input)
                if symptoms_response:
                    response = symptoms_response
                    st.session_state.suggested_disease = suggested_disease
                else:
                    # For other queries, use Mistral AI
                    response = chat_with_mistral(user_input)
    
    # Add AI response to conversation
    st.session_state.conversation.append({"role": "assistant", "content": response})
    
    return response

# **8️⃣ Streamlit UI Setup**
# Initialize session state for messages if not already set
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for information
st.sidebar.header("About this App")
st.sidebar.write("""
This Medical AI Assistant can:
- Screen for common diseases
- Provide health information
- Find healthcare facilities

*Note: This application is for educational purposes only and should not replace professional medical advice.*
""")

st.sidebar.header("Available Tests")
st.sidebar.write("""
- Diabetes
- Heart Disease
- Parkinson's
- Liver Disease
- Kidney Disease
""")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Type health your message..."):
    # Append user's message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Process response (Placeholder for AI response logic)
    response = "Hello! I'm here to help you with general health-related questions. How can I assist you today? 😊"
    
    # Append bot's response if it's not already appended
    if not (st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant" and st.session_state.messages[-1]["content"] == response):
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


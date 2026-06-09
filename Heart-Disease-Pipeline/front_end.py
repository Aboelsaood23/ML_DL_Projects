import streamlit as st
import requests
import pandas as pd
import os

# 1. Page Configuration & Styling
st.set_page_config(
    page_title="Multi-Model Cardiac Risk Inference Engine",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Multi-Model Cardiac Risk Inference Engine")
st.markdown("---")

# Backend FastAPI URL endpoint
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000/predict")

# 2. Setup Responsive UI Columns for Input Form
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Patient Demographics & Vitals")
    age = st.slider("Patient Age", min_value=1, max_value=120, value=55)
    sex_label = st.radio("Gender / Biological Sex", ["Male", "Female"])
    sex = 1 if sex_label == "Male" else 0
    
    resting_blood_pressure = st.slider("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=130)
    cholesterol = st.slider("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
    
    fbs_label = st.radio("Fasting Blood Sugar > 120 mg/dl", ["False / Normal", "True / Elevated"])
    fasting_blood_sugar = 1 if fbs_label == "True / Elevated" else 0

with col2:
    st.subheader("🔬 Clinical & Diagnostic Metrics")
    max_heart_rate = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    
    exang_label = st.radio("Exercise Induced Angina (Chest Pain)", ["No", "Yes"])
    exercise_induced_angina = 1 if exang_label == "Yes" else 0
    
    oldpeak = st.slider("ST Depression Induced by Exercise (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    vessels_colored = st.slider("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)
    
    chest_pain_type = st.selectbox(
        "Chest Pain Type Classification", 
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain"]
    )
    
    resting_ecg = st.selectbox("Resting Electrocardiographic Results", ["0", "1"])
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Flat", "Upsloping"])
    thalassemia = st.selectbox("Thalassemia Diagnostic Type", ["Normal", "Reversible Defect", "Unknown/Null"])

st.markdown("---")

# 3. Handle Submit and Process Request Response Loop
if st.button("Run Cardiac Risk Evaluation", use_container_width=True):
    
    # Pack the user interface widgets into the exact payload expected by app.py
    payload = {
        "age": age,
        "sex": sex,
        "resting_blood_pressure": float(resting_blood_pressure),
        "cholesterol": float(cholesterol),
        "fasting_blood_sugar": fasting_blood_sugar,
        "max_heart_rate": float(max_heart_rate),
        "exercise_induced_angina": exercise_induced_angina,
        "oldpeak": float(oldpeak),
        "chest_pain_type": chest_pain_type,
        "resting_ecg": resting_ecg,
        "slope": slope,
        "vessels_colored": vessels_colored,
        "thalassemia": thalassemia
    }
    
    with st.spinner("Broadcasting request across multi-model pipeline..."):
        try:
            # Post the JSON data package across the local port to FastAPI
            response = requests.post(FASTAPI_URL, json=payload)
            
            if response.status_code == 200:
                result_data = response.json()
                predictions_list = result_data.get("predictions", [])
                
                # Convert the returned list into a Pandas DataFrame for presentation
                df_results = pd.DataFrame(predictions_list)
                
                # Re-index/rename columns slightly for professional visual output
                df_results.columns = ["Machine Learning Model", "Raw Label", "Clinical Assessment", "Risk Confidence Score"]
                
                st.success("✅ Assessment Finished Successfully!")
                
                # 4. Highlight Panel Results using Status Metrics
                # Check consensus by looking at the first model's text assessment
                consensus_assessment = df_results["Clinical Assessment"].iloc[0]
                
                if "High Risk" in consensus_assessment:
                    st.error("CONSENSUS NOTICE: SYSTEM FLAGS HIGH CARDIOVASCULAR RISK PATHWAYS")
                else:
                    st.success("CONSENSUS NOTICE: PATIENT PROFILE BALANCED / LOW RISK MARKERS")
                
                # Display beautiful dataframe dashboard
                st.subheader("📊 Comparative Multi-Model Diagnostic Matrix")
                st.dataframe(
                    df_results.style.format({"Risk Confidence Score": "{:.2%}"}),
                    use_container_width=True,
                    hide_index=True
                )
                
            else:
                st.error(f"Backend server returned an operational error code: {response.status_code}")
                st.json(response.json())
                
        except requests.exceptions.ConnectionError:
            st.error("Network Pipeline Broken: Could not connect to the FastAPI backend! Is your Uvicorn server running?")
import pandas as pd 
import joblib 
from pathlib import Path

# Setup paths dynamically relative to this file's location
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
MODELS_DIR = PROJECT_ROOT / 'models'

#defin preprcessing function for API 
def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Takes a single patient dictionary from FastAPI, applies one-hot encoding 
    alignments, loads the saved training scaler, and outputs a clean 18-column row.
    """
    # Convert input dict to DataFrame
    df_raw = pd.DataFrame([data])
    #define all columns we model should receive 
    # 1. THE EXACT 18 TRAINING COLUMNS (Copied straight from your model)
    production_schema = [
        'Patient_Age', 'Gender', 'Resting_Blood_Pressure', 'Serum_Cholestoral', 'Fasting_Blood_Sugar', 
        'Resting_Electrocardiographic', 'Maximum_Heart_Rate', 'Exercise_Induced_Angina', 'ST_depression_induced_exercise', 
        'Major_vessels_colored', 'Chest_Pain_Type_Atypical Angina', 'Chest_Pain_Type_Non-anginal Pain', 
        'Chest_Pain_Type_Typical Angina', 'Slope_peak_exercise_Flat', 'Slope_peak_exercise_Upsloping', 
        'thal_Normal', 'thal_Reversible Defect', 'thal_Unknown/Null'
    ]
    
    # 2. Initiate empty dataframe with all columns set to 0.0 float
    df_engineered = pd.DataFrame(0.0, index=[0], columns=production_schema)
    
    # 3. Explicitly map direct numerical/binary inputs onto model columns
    df_engineered.at[0, 'Patient_Age'] = float(df_raw.at[0, 'age'])
    df_engineered.at[0, 'Gender'] = float(df_raw.at[0, 'sex'])
    df_engineered.at[0, 'Resting_Blood_Pressure'] = float(df_raw.at[0, 'resting_blood_pressure'])
    df_engineered.at[0, 'Serum_Cholestoral'] = float(df_raw.at[0, 'cholesterol'])
    df_engineered.at[0, 'Fasting_Blood_Sugar'] = float(df_raw.at[0, 'fasting_blood_sugar'])
    try:
        df_engineered.at[0, 'Resting_Electrocardiographic'] = float(df_raw.at[0, 'resting_ecg'])
    except ValueError:
        df_engineered.at[0, 'Resting_Electrocardiographic'] = 0.0
        
    df_engineered.at[0, 'Maximum_Heart_Rate'] = float(df_raw.at[0, 'max_heart_rate'])
    df_engineered.at[0, 'Exercise_Induced_Angina'] = float(df_raw.at[0, 'exercise_induced_angina'])
    df_engineered.at[0, 'ST_depression_induced_exercise'] = float(df_raw.at[0, 'oldpeak'])
    df_engineered.at[0, 'Major_vessels_colored'] = float(df_raw.at[0, 'vessels_colored'])

    # Chest Pain Type (Capitalized prefix matches: 'Chest_Pain_Type_')
    cp_col = f"Chest_Pain_Type_{df_raw.at[0, 'chest_pain_type']}"
    if cp_col in df_engineered.columns:
        df_engineered.at[0, cp_col] = 1.0
        
    # Slope (Prefix matches: 'Slope_peak_exercise_')
    slope_col = f"Slope_peak_exercise_{df_raw.at[0, 'slope']}"
    if slope_col in df_engineered.columns:
        df_engineered.at[0, slope_col] = 1.0
        
    # Thalassemia (Prefix matches: 'thal_')
    thal_col = f"thal_{df_raw.at[0, 'thalassemia']}"
    if thal_col in df_engineered.columns:
        df_engineered.at[0, thal_col] = 1.0

    # 5. Load saved training scaler and transform
    scaler_path = MODELS_DIR / 'StandardScaler_model.joblib' 
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        
        # Isolate exactly the 5 columns the scaler expects
        scaler_features = [
            'Patient_Age', 
            'Resting_Blood_Pressure', 
            'Serum_Cholestoral', 
            'Maximum_Heart_Rate', 
            'ST_depression_induced_exercise'
        ]
        df_to_scale = df_engineered[scaler_features]
        
        # Scale just those 5 numerical columns
        scaled_array = scaler.transform(df_to_scale)
        
        # Overwrite only those 5 columns in our main dataframe with the scaled values
        df_engineered[scaler_features] = scaled_array
        
        # Return the complete, clean 18-column dataframe to app.py
        return df_engineered
    else:
        raise FileNotFoundError(f"Scaler file missing from: {scaler_path}")
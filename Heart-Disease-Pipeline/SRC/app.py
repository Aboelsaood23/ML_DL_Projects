from SRC.preprocessing import preprocess_input
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib

#Initialize FastAPI app
app = FastAPI(
    title = 'Heart Disease Inference Server'
) 

# Setup path routes relative to project architecture
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
MODELS_DIR = PROJECT_ROOT / 'models'

#dictinary to hold all models 
loaded_models = {}

model_files = {
    "Decision Tree": "Decision_Tree_model.joblib",
    "K-Nearest Neighbors": "KNN_model.joblib",
    "Logistic Regression": "Logistic_Regression_model.joblib",
    "Naive Bayes": "Naive_Bayes_model.joblib",
    "Random Forest": "Random_Forest_model.joblib",
    "Support Vector Machine": "SVM_model.joblib",
    "XGBoost": "XGBoost_model.joblib"
}


# Load all available models once at server startup
print("Loading inference engines into memory...")
for model_name, file_name in model_files.items():
    full_path = MODELS_DIR / file_name
    if full_path.exists():
        loaded_models[model_name] = joblib.load(full_path)
        print(f"Loaded: {model_name}")
    else:
        print(f"Skipping (Not Found): {model_name}")

if not loaded_models:
    raise RuntimeError("No classification models were successfully loaded!")

#Define pydantc patiant data 
class PatientFeatures(BaseModel):
    age: int
    sex: int                     # 1 = Male, 0 = Female
    resting_blood_pressure: float # trestbps
    cholesterol: float            # chol
    fasting_blood_sugar: int     # fbs (1 = true; 0 = false)
    max_heart_rate: float        # thalach
    exercise_induced_angina: int # exang (1 = yes; 0 = no)
    oldpeak: float               # ST depression induced by exercise
    chest_pain_type: str         # e.g., "Typical Angina", "Asymptomatic"
    resting_ecg: str             # e.g., "Normal", "ST-T wave abnormality"
    slope: str                   # e.g., "Upsloping", "Flat"
    vessels_colored: int         # ca (0-3)
    thalassemia: str             # e.g., "Normal", "Fixed Defect"

# The Root Door ("/")
@app.get("/")
def home():
    return {
        "status": "Online", 
        "available_engines": list(loaded_models.keys())
    }



@app.post("/predict")
def predict_multi_model(patient: PatientFeatures):
    try:
        patient_dict = patient.dict()
        #Preprocess and scale the incoming row
        df_engineered = preprocess_input(patient_dict)
        #Loop through all loaded models and compile individual results
        model_results = []
        for name, model in loaded_models.items():
            # Get 0 or 1 prediction
            binary_pred = int(model.predict(df_engineered)[0])
            risk_label = "High Risk" if binary_pred == 0 else "Low Risk / Normal"
            # Safely handle confidence scores if the model supports probabilities
            if hasattr(model, "predict_proba"):
                prob_array = model.predict_proba(df_engineered)[0]
                confidence = round(float(prob_array[0]), 4)
            else:
                # Fallback for models without predict_proba (like linear SVM configuration)
                confidence = "N/A"
                
            model_results.append({
                "model_name": name,
                "prediction": binary_pred,
                "assessment": risk_label,
                "confidence_score": confidence
            })         
        #Return the array containing results for all models
        return {
            "status": "Success",
            "total_models_evaluated": len(model_results),
            "predictions": model_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-Inference Error: {str(e)}")
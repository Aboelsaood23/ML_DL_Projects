import logging
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#create the logging directory and configure logging
PROJECT_ROOT = Path(__file__).resolve().parent.parent
log_dir = PROJECT_ROOT / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'pipeline.log'),
        logging.StreamHandler()
    ]
)

#multimodel blueprint configuration
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Raw Benchmarks
models_blueprint = {
    'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive_Bayes': GaussianNB(),
    'Decision_Tree': DecisionTreeClassifier(random_state=42),
    'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss') 
}

# GridSearch Search Spaces
svm_param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 4, 6, 10], 'class_weight': ['balanced', None]}
xgb_param_grid = {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 4, 6]}

tuned_blueprint = {
    'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive_Bayes': GaussianNB(),
    'Decision_Tree': DecisionTreeClassifier(random_state=42),
    'SVM': joblib.Parallel(n_jobs=-1)(delayed(lambda: None)()) if False else None, # Handled dynamically below
}

# Modeling and evaluation pipeline
def execute_training_pipeline(models_dict, x_train, x_test, y_train, y_test, run_tuning=False):
    performance_matrix = {}
    
    # Dynamically handle model setups based on whether we are tuning or baseline testing
    active_blueprint = {}
    from sklearn.model_selection import GridSearchCV
    
    for name, model in models_blueprint.items():
        if run_tuning and name == 'SVM':
            active_blueprint[name] = GridSearchCV(SVC(probability=True, random_state=42), svm_param_grid, cv=cv_strategy, scoring='roc_auc', n_jobs=-1)
        elif run_tuning and name == 'Random_Forest':
            active_blueprint[name] = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=cv_strategy, scoring='roc_auc', n_jobs=-1)
        elif run_tuning and name == 'XGBoost':
            active_blueprint[name] = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), xgb_param_grid, cv=cv_strategy, scoring='roc_auc', n_jobs=-1)
        else:
            active_blueprint[name] = model

    for model_name, model_obj in active_blueprint.items():
        logging.info(f"Processing model: {model_name}")
        try:
            model_obj.fit(x_train, y_train)
            
            if hasattr(model_obj, 'best_params_'):
                logging.info(f"[{model_name}] Optimal Hyperparameters: {model_obj.best_params_}")
                active_model = model_obj.best_estimator_
            else:
                active_model = model_obj
                
            y_pred = active_model.predict(x_test)
            y_prob = active_model.predict_proba(x_test)[:, 1] if hasattr(active_model, "predict_proba") else None
            
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1-Score': f1_score(y_test, y_pred, zero_division=0),
                'ROC_AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
            }
            performance_matrix[model_name] = metrics
            
            # Save tuned estimators to artifacts directory ONLY during the final tuning execution loop
            if run_tuning:
                models_dir = PROJECT_ROOT / 'models'
                models_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(active_model, models_dir / f"{model_name}_model.joblib")

        except Exception as e:
            logging.error(f"Execution crash model {model_name}. Error details: {str(e)}")

    return pd.DataFrame(performance_matrix).T.sort_values(by='Recall', ascending=False)

# Excution
if __name__ == "__main__":
    logging.info("Production Model Training Script Initialized.")
    
    # Set up data paths relative to script location
    data_dir = PROJECT_ROOT / 'Data' / 'Final'
    output_dir = PROJECT_ROOT / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load feature blocks cleanly
    x_train = pd.read_csv(data_dir / "x_train_scaled.csv")
    x_test = pd.read_csv(data_dir / "x_test_scaled.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
    
    logging.info(f"Dimensions verified: x_train {x_train.shape} | x_test {x_test.shape}")
    
    #Baseline Execution Loop
    logging.info("Starting Run 1: Generating Raw Baselines Matrix...")
    df_baseline_order = execute_training_pipeline(models_blueprint, x_train, x_test, y_train, y_test, run_tuning=False)
    df_baseline_order.to_csv(output_dir / "baseline_order.csv", index=True)
    logging.info("Baseline performance matrix saved to /output/baseline_order.csv")
    
    #Advanced Tuning & Artifact Preservation Loop
    logging.info("Starting Run 2: Executing Hyperparameter Optimization and Serialization...")
    df_best_estimator = execute_training_pipeline(models_blueprint, x_train, x_test, y_train, y_test, run_tuning=True)
    df_best_estimator.to_csv(output_dir / "best_estimator.csv", index=True)
    logging.info("Tuned performance matrix saved to /output/best_estimator.csv")
    
    logging.info("Script execution successfully completed. All assets secured.")
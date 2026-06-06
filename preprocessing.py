import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ clen input data , duplicate values and missing values """
    df = df.copy()
    #remove duplicates 
    df = df.drop_duplicates()
    # Strip leading and trailing whitespace from column names
    df.columns = df.columns.str.strip()
    return df 

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """ Handel Nana and missing hidden values """
    df = df.copy()
    # Replace placeholders like '?'
    df = df.replace("?", np.nan)
    df = df.replace(" ", np.nan)
    # Fill numeric NaNs with median (safe default for medical data)
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    # Fill categorical columns with mode or 'Unknown'
    categorical_cols = df.select_dtypes(include="object").columns
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """" Map categorical varibles and one-hot encoding"""
    df = df.copy()
    chest_pain_map = {
        0: 'Typical Angina',
        1: 'Atypical Angina',
        2: 'Non-anginal Pain',
        3: 'Asymptomatic'
    }

    slope_map = {
        0: 'Upsloping',
        1: 'Flat',
        2: 'Downsloping'
    }

    thal_map = {
        0: 'Unknown/Null',
        1: 'Normal',
        2: 'Fixed Defect',
        3: 'Reversible Defect'
    }

    df['Chest_Pain_Type'] = df['Chest_Pain_Type'].map(chest_pain_map)
    df['Slope_peak_exercise'] = df['Slope_peak_exercise'].map(slope_map)
    df['thal'] = df['thal'].map(thal_map)

    df = pd.get_dummies(df,
                        columns=['Chest_Pain_Type', 'Slope_peak_exercise', 'thal'],
                        drop_first=True,
                        dtype=int)
    return df

def split_data(df: pd.DataFrame, target_col: str = 'target'):
    """split dataset into train/test"""
    x = df.drop(columns=[target_col])
    y = df[target_col]
    x_train,x_test,y_train,y_test = train_test_split(
        x,y,test_size = 0.2,stratify=y,random_state=42
    )
    return x_train,x_test,y_train,y_test

def scale_data(x_train,x_test,continuous_features):
    """ Apply StandardScaler after doing split to avoid data leakafe"""
    scaler = StandardScaler()
    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train[continuous_features] = scaler.fit_transform( x_train[continuous_features])

    x_test[continuous_features] = scaler.transform(
        x_test[continuous_features]
    )

    return x_train, x_test, scaler

#define full pipline 
def preprocess_pipline(df:pd.DataFrame):
    """Full preprocessing pipline"""
    df = clean_data(df)
    df = handle_missing_values(df)
    df = encode_features(df)
    x_train,x_test,y_train,y_test = split_data(df)
    continuous_features = [
        'Patient_Age',
        'Resting_Blood_Pressure',
        'Serum_Cholestoral',
        'Maximum_Heart_Rate',
        'ST_depression_induced_exercise'
    ]

    x_train, x_test, scaler = scale_data(x_train, x_test, continuous_features)

    return x_train, x_test, y_train, y_test, scaler

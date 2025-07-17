# utils/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

def preprocess_data(df: pd.DataFrame, target: str):
    df = df.copy()

    # Drop columns with >50% missing
    null_threshold = 0.5
    df = df.loc[:, df.isnull().mean() < null_threshold]

    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Detect task type
    if y.dtype == 'object' or y.nunique() <= 10:
        task_type = 'classification'
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        task_type = 'regression'

    # Identify columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    # Pipelines
    num_pipeline = Pipeline(steps=[
        ("imputer", KNNImputer(n_neighbors=3)),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # Final train-test split
    X_processed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    st.success(f"Preprocessing done. Task type detected: {task_type.capitalize()}")

    return X_train, X_test, y_train, y_test, task_type

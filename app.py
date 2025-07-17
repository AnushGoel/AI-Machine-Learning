# app.py

import streamlit as st
import pandas as pd
from utils.eda import run_eda
from utils.graphing import run_graphs
from utils.preprocessing import preprocess_data
from utils.modeling import run_model

st.set_page_config(page_title="AutoML Explorer", layout="wide")

st.title("📊 AutoML & EDA Dashboard")

# Step 1: Upload
file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if file:
    df = pd.read_csv(file)
    st.success("File uploaded successfully!")
    st.dataframe(df.head())

    # Step 2: Select Target
    target = st.selectbox("Select the target column (Y variable)", df.columns)

    if target:
        # Step 3: Run EDA
        st.subheader("🔍 Exploratory Data Analysis")
        run_eda(df, target)

        # Step 4: Graphs
        st.subheader("📈 Visual Analysis")
        run_graphs(df, target)

        # Step 5: Preprocess
        st.subheader("⚙️ Data Preprocessing")
        X_train, X_test, y_train, y_test, task_type = preprocess_data(df, target)

        # Step 6: Modeling
        st.subheader("🤖 ML Modeling & Evaluation")
        run_model(X_train, X_test, y_train, y_test, task_type)

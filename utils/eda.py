# utils/eda.py

import pandas as pd
import streamlit as st

def run_eda(df: pd.DataFrame, target: str):
    st.markdown("### 📌 Dataset Overview")
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())

    st.markdown("### 🔍 Data Types")
    st.dataframe(df.dtypes)

    st.markdown("### 🧼 Missing Values")
    null_df = df.isnull().sum().reset_index()
    null_df.columns = ["Column", "Missing Values"]
    null_df["% Missing"] = round(null_df["Missing Values"] / len(df) * 100, 2)
    st.dataframe(null_df[null_df["Missing Values"] > 0])

    st.markdown("### 📊 Descriptive Stats")
    st.dataframe(df.describe(include='all').transpose())

    st.markdown("### 🎯 Target Variable Distribution")
    st.write(df[target].value_counts(normalize=True) if df[target].dtype == "object" else df[target].describe())

# utils/graphing.py

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

def run_graphs(df: pd.DataFrame, target: str):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.markdown("### 📊 Univariate Plots")

    # Numeric - Histogram
    st.write("**Histogram (Numeric Columns)**")
    for col in numeric_cols:
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig, use_container_width=True)

    # Categorical - Bar Plot
    st.write("**Bar Plots (Categorical Columns)**")
    for col in categorical_cols:
        fig = px.bar(df[col].value_counts().reset_index(), x="index", y=col, labels={"index": col, col: "Count"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📈 Bivariate Plots")

    # Numeric vs Target
    if target in numeric_cols:
        for col in numeric_cols:
            if col != target:
                fig = px.scatter(df, x=col, y=target)
                st.plotly_chart(fig, use_container_width=True)
    else:
        for col in numeric_cols:
            fig = px.box(df, x=target, y=col)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔥 Correlation Heatmap (Numeric Only)")
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

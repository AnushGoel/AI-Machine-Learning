import streamlit as st
import pandas as pd

from utils.eda import run_eda
from utils.modeling import run_modeling
from utils.graphing import plot_histograms, plot_boxplots, plot_pairplot
from utils.preprocessing import auto_preprocess

st.set_page_config(page_title="AutoML Dashboard", layout="wide")

st.title("🤖 AutoML System with EDA, Modeling & Visualization")

# --- File Upload ---
st.sidebar.header("📁 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File uploaded successfully")

        # Target column
        target = st.sidebar.selectbox("🎯 Select Target Variable", df.columns)

        # Auto preprocessing
        df = auto_preprocess(df)

        # --- Tabs for App Sections ---
        tabs = st.tabs(["📊 EDA", "📈 Graphs", "⚙️ Modeling"])

        # --- EDA Tab ---
        with tabs[0]:
            try:
                run_eda(df, target)
            except Exception as e:
                st.error(f"❌ EDA Error: {e}")

        # --- Graphs Tab ---
        with tabs[1]:
            try:
                plot_histograms(df)
                plot_boxplots(df)
                plot_pairplot(df, target)
            except Exception as e:
                st.error(f"❌ Graphing Error: {e}")

        # --- Modeling Tab ---
        with tabs[2]:
            try:
                run_modeling(df, target)
            except Exception as e:
                st.error(f"❌ Modeling Error: {e}")

    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
else:
    st.warning("📤 Please upload a CSV file to begin.")

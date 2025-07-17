# utils/modeling.py

import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_squared_error, r2_score
from lazypredict.Supervised import LazyClassifier, LazyRegressor

def run_model(X_train, X_test, y_train, y_test, task_type):
    st.write("Running models... please wait ⏳")

    if task_type == "classification":
        clf = LazyClassifier(verbose=0, ignore_warnings=True)
        models, _ = clf.fit(X_train, X_test, y_train, y_test)
        st.subheader("📋 Model Comparison (Classification)")
        st.dataframe(models)

        best_model_name = models.index[0]
        st.success(f"Top Model: {best_model_name}")

    else:
        reg = LazyRegressor(verbose=0, ignore_warnings=True)
        models, _ = reg.fit(X_train, X_test, y_train, y_test)
        st.subheader("📋 Model Comparison (Regression)")
        st.dataframe(models)

        best_model_name = models.index[0]
        st.success(f"Top Model: {best_model_name}")

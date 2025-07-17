# utils/modeling.py

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from math import sqrt
import warnings

warnings.filterwarnings("ignore")

def run_model(X_train, X_test, y_train, y_test, task_type):
    st.write("Running models... please wait ⏳")
    results = {}

    if task_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            "Naive Bayes": GaussianNB(),
            "AdaBoost": AdaBoostClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average='weighted')
                results[name] = {"Accuracy": round(acc, 4), "F1 Score": round(f1, 4)}
            except Exception as e:
                results[name] = {"Accuracy": "Error", "F1 Score": f"⚠️ {str(e)[:30]}"}

        result_df = pd.DataFrame(results).T
        result_df = result_df.sort_values("Accuracy", ascending=False, na_position='last')
        st.write("### 📋 Model Performance (Classification)")
        st.dataframe(result_df)

        best_model_name = result_df.index[0]
        st.success(f"✅ Best Model: {best_model_name}")

        try:
            best_model = models[best_model_name]
            preds = best_model.predict(X_test)
            st.write("#### 🔮 Sample Predictions")
            st.write(preds[:10])
        except:
            st.warning("Best model failed during prediction.")

    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "SVR": SVR(),
            "AdaBoost": AdaBoostRegressor(),
            "XGBoost": XGBRegressor()
        }

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                rmse = sqrt(mean_squared_error(y_test, preds))
                results[name] = {"R² Score": round(r2, 4), "RMSE": round(rmse, 4)}
            except Exception as e:
                results[name] = {"R² Score": "Error", "RMSE": f"⚠️ {str(e)[:30]}"}

        result_df = pd.DataFrame(results).T
        result_df = result_df.sort_values("R² Score", ascending=False, na_position='last')
        st.write("### 📋 Model Performance (Regression)")
        st.dataframe(result_df)

        best_model_name = result_df.index[0]
        st.success(f"✅ Best Model: {best_model_name}")

        try:
            best_model = models[best_model_name]
            preds = best_model.predict(X_test)
            st.write("#### 🔮 Sample Predictions")
            st.write(preds[:10])
        except:
            st.warning("Best model failed during prediction.")

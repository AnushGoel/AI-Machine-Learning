# utils/modeling.py

import streamlit as st
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

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
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')
            results[name] = {"Accuracy": acc, "F1 Score": f1}

        result_df = (
            pd.DataFrame(results).T
            .sort_values("Accuracy", ascending=False)
        )
        best_model_name = result_df.index[0]
        st.write("### 🔍 Model Performance (Classification)")
        st.dataframe(result_df)
        st.success(f"✅ Best Model: {best_model_name}")

        # Predict again with best
        best_model = models[best_model_name]
        preds = best_model.predict(X_test)
        st.write("#### 🔮 Sample Predictions")
        st.write(preds[:10])

    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "SVR": SVR(),
            "AdaBoost": AdaBoostRegressor(),
            "XGBoost": XGBRegressor()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            rmse = mean_squared_error(y_test, preds, squared=False)
            results[name] = {"R² Score": r2, "RMSE": rmse}

        result_df = (
            pd.DataFrame(results).T
            .sort_values("R² Score", ascending=False)
        )
        best_model_name = result_df.index[0]
        st.write("### 📊 Model Performance (Regression)")
        st.dataframe(result_df)
        st.success(f"✅ Best Model: {best_model_name}")

        best_model = models[best_model_name]
        preds = best_model.predict(X_test)
        st.write("#### 🔮 Sample Predictions")
        st.write(preds[:10])

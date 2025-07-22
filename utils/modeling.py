import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

def run_modeling(df, target):
    st.markdown("## 🤖 Model Training & Evaluation")

    try:
        X = df.drop(columns=[target])
        y = df[target]

        # Encode categorical if any
        X = pd.get_dummies(X)

        # Classification or regression
        problem_type = 'classification' if y.nunique() <= 10 and y.dtype in ['int64', 'object'] else 'regression'
        st.info(f"Detected problem type: **{problem_type}**")

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model options
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier() if problem_type == "classification" else RandomForestRegressor(),
            "SVM": SVC(probability=True) if problem_type == "classification" else SVR(),
            "XGBoost": xgb.XGBClassifier(eval_metric='logloss') if problem_type == "classification" else xgb.XGBRegressor()
        }

        selected_model = st.selectbox("Choose a model", list(models.keys()))
        model = models[selected_model]

        # Train
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        st.success(f"{selected_model} training completed.")

        # Evaluation
        if problem_type == "classification":
            st.write("### 📈 Accuracy:", accuracy_score(y_test, y_pred))
            st.write("### 📄 Classification Report")
            st.text(classification_report(y_test, y_pred))

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            st.write("### 🔍 Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            # ROC Curve
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                st.write("### 🔵 ROC Curve")
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax2.plot([0, 1], [0, 1], 'k--')
                ax2.set_xlabel('False Positive Rate')
                ax2.set_ylabel('True Positive Rate')
                ax2.legend()
                st.pyplot(fig2)

        else:
            # Regression metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            st.write(f"### 📏 RMSE: {rmse:.2f}")
            st.write(f"### 🎯 R² Score: {r2:.2f}")

        # Feature Importance (if available)
        if hasattr(model, "feature_importances_"):
            st.write("### 📌 Feature Importance")
            importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            fig3, ax3 = plt.subplots()
            sns.barplot(x="Importance", y="Feature", data=importance_df.head(15), ax=ax3)
            st.pyplot(fig3)

    except Exception as e:
        st.error(f"❌ Error during modeling: {e}")

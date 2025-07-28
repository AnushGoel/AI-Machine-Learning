
# 🚀 AutoML Web App – Explore, Train, and Predict with Ease!

[![Streamlit App](https://img.shields.io/badge/Launch-App-green?style=for-the-badge)](https://ppc7dpe5rsae8sfwl6lx72.streamlit.app/)

## 🌟 Overview

**AutoML Web App** is an end-to-end interactive machine learning platform built using **Streamlit**. It empowers users to upload their datasets, perform **EDA**, **visualize**, **preprocess**, **train models**, and **make predictions** — all in a few clicks, without writing a single line of code!

## 🧠 Key Features

- ✅ Upload CSV datasets  
- ✅ Automatic EDA report generation  
- ✅ Interactive visualizations  
- ✅ Smart preprocessing (handle nulls, encoding, scaling)  
- ✅ Model training with best-fit algorithm (Classification or Regression)  
- ✅ Compare model performance  
- ✅ Make live predictions  
- ✅ Lightweight and easy to deploy via Streamlit

## 🔍 How It Works

1. **Upload Dataset:** Choose any CSV file.
2. **Explore:** View statistical summaries and data health.
3. **Visualize:** Generate histograms, pair plots, and correlation heatmaps.
4. **Train Models:** Automatically detects task (regression/classification), selects algorithms, and evaluates performance.
5. **Predict:** Use your trained model to make predictions on new data.

## 🛠️ Tech Stack

- Frontend: [Streamlit](https://streamlit.io/)
- Backend: Python
- Libraries: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, etc.
- Architecture: Modularized code in `utils/` for clarity and reuse

## 📦 Installation

```bash
git clone https://github.com/yourusername/AutoML-WebApp.git
cd AutoML-WebApp
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Folder Structure

```
.
├── app.py                 # Streamlit frontend
├── requirements.txt       # Python dependencies
└── utils/                 # All core logic modules
    ├── eda.py
    ├── graphing.py
    ├── modeling.py
    ├── predictor.py
    └── preprocessing.py
```


## 🧪 Sample Usage

Try it live:  
👉 [Streamlit App](https://ppc7dpe5rsae8sfwl6lx72.streamlit.app/)

Use a sample CSV like:

```
Name,Age,Salary,Target
John,32,50000,Yes
...
```

## 🧠 Suggested Better Name

- **AutoML-WebApp**
- **SmartML Dashboard**
- **NoCodeML**
- **Click2Predict**
- **MLPilot**

## 📜 License

MIT License

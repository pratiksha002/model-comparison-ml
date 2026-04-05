# 🚀 End-to-End Car Price Prediction System

## 🧠 Overview

This project is a complete end-to-end Machine Learning system that predicts car selling prices using real-world data. It covers the full ML lifecycle—from preprocessing and model training to explainability and API deployment.

---

## 🎯 Objectives

- Build a reusable and automated preprocessing pipeline  
- Compare multiple machine learning models  
- Select the best-performing model  
- Explain model predictions  
- Deploy the model using an API for real-time predictions  

---

## 🏗️ Project Architecture

Data → Preprocessing → Model Training → Model Comparison → Best Model → Explainability → API Deployment

---

## ⚙️ Features

### 🔹 1. Custom Preprocessing Pipeline
- Missing value handling (median & most frequent)
- Outlier handling (IQR clipping)
- Skewness correction (log transformation)
- Feature scaling (StandardScaler)
- Target encoding for categorical variables
- Feature selection (Variance Threshold)

---

### 🔹 2. Model Comparison
Multiple models are trained and compared:

- Ridge Regression  
- Random Forest Regressor  
- XGBoost Regressor  

Best model is selected based on **R² score**.

---

### 🔹 3. Model Explainability
- Feature Importance (global explanation)
- SHAP (local explanation for individual predictions)

---

### 🔹 4. API Deployment (FastAPI)
- Real-time predictions via API  
- Input validation using Pydantic  
- Structured JSON responses  
- Error handling  

---

## 📊 Sample API Input

``json
{
  "Year": 2015,
  "Present_Price": 5.5,
  "Kms_Driven": 50000,
  "Fuel_Type": "Petrol",
  "Seller_Type": "Dealer",
  "Transmission": "Manual",
  "Owner": 0
}

---

##📤 Sample API Output
---
{
  "success": true,
  "prediction": 3.51,
  "model_used": "XGBoost",
  "message": "Prediction generated successfully"
}
----
##🛠️ Tech Stack
Python
Pandas, NumPy
Scikit-learn
XGBoost
FastAPI
SHAP
Joblib

----
model_comparison_project/
│
├── data/
│   └── car_data.csv
│
├── src/
│   ├── preprocessor.py
│   ├── models.py
│   ├── explain.py
│   ├── shap_explain.py
│
├── train.py
├── app.py
├── models/
│   └── best_model.pkl
├── requirements.txt
└── README.md

---
##▶️ How to Run
1. Clone the repository
git clone <your-repo-link>
cd model_comparison_project
2. Install dependencies
pip install -r requirements.txt
3. Train the model
python train.py
4. Run the API
uvicorn app:app --reload
5. Access API Docs

---
Open in browser:

http://127.0.0.1:8000/docs

---
🧠 Key Learnings
Building end-to-end ML pipelines
Avoiding data leakage using pipelines
Importance of model comparison
Understanding model behavior with explainability
Deploying ML models as APIs

---
🚀 Future Improvements
Add frontend UI (React / Streamlit)
Deploy API to cloud (Render / AWS)
Add logging & monitoring
Use larger datasets for better generalization

---
📌 Conclusion

This project demonstrates how to transition from building ML models to deploying complete ML systems that are usable in real-world applications.

⭐ If you found this useful

Give this repo a ⭐ and feel free to connect!

from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/best_model.pkl")

@app.get("/")
def home():
    return{"message": "ML Model API is running"}

@app.post("/predict")
def predict(data: dict):
    try:
        input_data = pd.DataFrame([data])

        prediction = model.predict(input_data)

        return {
        "success": True,
        "prediction": float(prediction[0]),
        "model_used": "XGBoost",
        "message": "Prediction generated successfully"
         }

    except Exception as e:
        return {"success": False,
                "error": str(e)
                }
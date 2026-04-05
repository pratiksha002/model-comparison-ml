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
        "prediction": float(prediction[0])
        }

    except Exception as e:
        return {"error": str(e)}
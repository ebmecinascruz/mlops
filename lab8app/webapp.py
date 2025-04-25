from fastapi import FastAPI, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
import joblib

# Input schema
class ObesityRequest(BaseModel):
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str

# Modern lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Loading model artifacts...")
    app.state.model = joblib.load("model_rf33.pkl")
    app.state.scaler = joblib.load("scaler.pkl")
    app.state.expected_columns = joblib.load("selected_feature_columns.pkl")
    app.state.numeric_columns = joblib.load("numeric_columns.pkl")
    print("✅ Model, scaler, and columns loaded.")
    yield
    print("🛑 Shutting down app...")

# Create app with lifespan
app = FastAPI(title="Obesity Classifier", version="2.0", lifespan=lifespan)

@app.post("/predict")
def predict(data: ObesityRequest, request: Request):
    print("🔥 FastAPI hit: /predict was called!")

    try:
        raw_dict = data.model_dump()
        print("📦 Raw input:", raw_dict)

        df = pd.DataFrame([raw_dict])
        print("🧾 Initial DataFrame:", df.to_dict(orient="records"))

        df_encoded = pd.get_dummies(df, columns=[
            "Gender", "family_history_with_overweight", "FAVC",
            "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"
        ])
        print("🧪 Encoded columns:", df_encoded.columns.tolist())

        # Add missing cols
        for col in request.app.state.expected_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        #df_encoded = df_encoded[request.app.state.expected_columns]
        missing = [col for col in request.app.state.expected_columns if col not in df_encoded.columns]
        print("❌ Missing columns NOT added:", missing)

        df_encoded = df_encoded.reindex(columns=request.app.state.expected_columns, fill_value=0)

        # Scale
        numeric_cols = request.app.state.numeric_columns
        df_encoded[numeric_cols] = request.app.state.scaler.transform(df_encoded[numeric_cols])

        print("🧪 Incoming cols:", set(df_encoded.columns))
        print("✅ Model expects:", set(request.app.state.expected_columns))
        # Predict
        prediction = request.app.state.model.predict(df_encoded)
        return {"prediction": prediction.tolist()}

    except Exception as e:
        return {"error": str(e)}
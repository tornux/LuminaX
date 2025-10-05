from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

model = joblib.load("gradient_boosting_model.pkl")

app = FastAPI(title="Gradient Boosting Model API")

# ✅ إضافة CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # أو ["http://127.0.0.1:5500"] لو هتشغل HTML من VS Code Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Features(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float
    feature6: float
    feature7: float
    feature8: float
    feature9: float
    feature10: float
    feature11: float
    feature12: float
    feature13: float
    feature14: float
    feature15: float
    feature16: float
    feature17: float
    feature18: float
    feature19: float
    feature20: float
    feature21: float
    feature22: float
    feature23: float
    feature24: float
    feature25: float
    feature26: float
    feature27: float
    feature28: float
    feature29: float
    feature30: float
    feature31: float
    feature32: float
    feature33: float
    feature34: float
    feature35: float
    feature36: float
    feature37: float
    feature38: float
    feature39: float
    feature40: float
    feature41: float
    feature42: float
    feature43: float

@app.post("/predict")
def predict(data: Features):
    X = [list(data.dict().values())]
    prediction = model.predict(X)[0]
    return {"prediction": float(prediction)}

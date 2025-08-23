from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

from app.model import predict_top_events, build_dashboard
from app.predictor import predict_safe_path

app = FastAPI(title="OrbitXOS API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    satellite_tle: str
    debris_tle: str
    horizon_minutes: Optional[int] = 60
    step_seconds: Optional[int] = 30

@app.get("/events")
def events(top_n: int = 6):
    return predict_top_events(top_n=top_n)

@app.get("/dashboard")
def dashboard():
    return build_dashboard()

@app.post("/predict")
def predict(req: PredictRequest = Body(...)):
    return predict_safe_path(
        satellite_tle=req.satellite_tle,
        debris_tle=req.debris_tle,
        horizon_minutes=req.horizon_minutes,
        step_seconds=req.step_seconds
    )

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


app = FastAPI(title="Satellite Risk Dashboard API")

@app.get("/dashboard")
def dashboard():
    """
    Homepage cards, recent alerts, high-priority tracking — dynamic.
    Returns the payload produced by build_dashboard.
    """
    return build_dashboard()


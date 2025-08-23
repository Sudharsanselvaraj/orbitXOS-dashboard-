from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.model import predict_top_events

app = FastAPI(title="🚀 Satellite Collision Risk API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API is running", "endpoints": ["/predict"]}

@app.get("/predict")
def predict():
    """Return top 4 most critical conjunction events."""
    return predict_top_events(top_n=4)

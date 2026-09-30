from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from schemas import PredictionRequest, DepthProfileRequest
from model_runner import predict, get_depth_profile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/predict")
def prediction(request: PredictionRequest):
    mean, std = predict(request.day_index)

    return {
        "day_index": request.day_index,
        "shape": list(mean.shape),
        "message": "Prediction generated successfully"
    }


@app.post("/api/v1/depth-profile")
def depth_profile(request: DepthProfileRequest):
    try:
        return get_depth_profile(
            request.day_index,
            request.latitude,
            request.longitude
        )
    except ValueError as e:
        # get_depth_profile raises ValueError for land points / out-of-domain
        # coordinates. Converted to a real HTTP error with a real message,
        # instead of an unhandled 500 "Internal Server Error."
        raise HTTPException(status_code=422, detail=str(e))


# Serve the frontend from the SAME server, same domain, so the deployed
# app is one link instead of two. Must be mounted AFTER the /api/v1/*
# routes above — StaticFiles with html=True catches "/" and everything
# else not already matched by a route defined earlier in this file.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
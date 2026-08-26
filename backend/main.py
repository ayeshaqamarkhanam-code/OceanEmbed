from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

@app.get("/")
def root():
    return {"message": "OceanEmbed backend is running"}


@app.post("/api/v1/predict")
def prediction(request: PredictionRequest):
    result = predict(request.day_index)

    return {
        "day_index": request.day_index,
        "shape": list(result.shape),
        "message": "Prediction generated successfully"
    }


@app.post("/api/v1/depth-profile")
def depth_profile(request: DepthProfileRequest):
    return get_depth_profile(
        request.day_index,
        request.latitude,
        request.longitude
    )
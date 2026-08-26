from pydantic import BaseModel


class PredictionRequest(BaseModel):
    day_index: int


class DepthProfileRequest(BaseModel):
    latitude: float
    longitude: float
    day_index: int = 0
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    day_index: int


class DepthProfileRequest(BaseModel):
    day_index: int
    latitude: float
    longitude: float
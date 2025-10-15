from pydantic import BaseModel
from typing import List, Optional

class ZonePrediction(BaseModel):
    zone_id: str
    p_flood_3h: float
    eta_minutes: float
    alert_level: Optional[str]
    confirmation_score: Optional[float]
    risk_score: Optional[float]
    rainfall_last_3h: Optional[float]
    river_level_m: Optional[float]
    soil_moisture: Optional[float]
    people_unable_to_evacuate: Optional[int]
    population_total: Optional[int]
    elevation_m: Optional[float]
    centroid_lat: Optional[float]
    centroid_lon: Optional[float]

class PredictionsResponse(BaseModel):
    timestamp: str
    model_version: Optional[str]
    zones: List[ZonePrediction]

class AlertZone(BaseModel):
    zone_id: str
    risk_score: float
    eta_minutes: float
    people_unable_to_evacuate: int
    centroid_lat: float
    centroid_lon: float

class AlertPayload(BaseModel):
    alert_id: str
    timestamp: str
    alert_type: str
    message: str
    zones: List[AlertZone]
    recommended_actions: List[str]

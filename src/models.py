from pydantic import BaseModel, Field, validator
from pathlib import Path
from typing import List, Optional, Dict, Any

class Metrics(BaseModel):
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    mAP50: float =  Field(ge=0.0, le=1.0)
    mAP50_95: float =  Field(ge=0.0, le=1.0)
    
    @validator('mAP50_95')
    def check_mAP(cls, v):
        return v
    
    
class ModelEpoch(BaseModel):
    epoch: int
    metrics: Metrics
    
class ModelTraining(BaseModel):
    model_name: str
    epoch_stage: int
    path: Path
    final_metrics: Metrics
    all_metrics: Dict[str, Any]
    dataset_source: str = "unknown"
    
    
from pydantic import BaseModel, Field
from typing import Dict

class ResponseModel(BaseModel):
    prediction: int = Field(..., description="Predicted placement status (0 or 1)", examples=[0, 1])
    probability: float = Field(..., description="Probability of being placed", examples=[0.75, 0.85])
    feature_contributions: Dict[str, float] = Field(..., description="Feature Importance in predicting output", examples=)
from pydantic import BaseModel, Field

class ResponseModel(BaseModel):
    prediction: int = Field(..., description="Predicted placement status (0 or 1)", examples=[0, 1])
    probability: float = Field(..., description="Probability of the predicted category", examples=[0.75, 0.85])
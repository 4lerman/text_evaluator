from pydantic import BaseModel, Field

class EvaluateRequest(BaseModel):
    """Request model for evaluation."""
    text: str = Field(..., description="The candidate's full answer to evaluate.")

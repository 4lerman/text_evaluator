from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import logging

from app.schemas.requests import EvaluateRequest
from app.schemas.responses import EvaluateResponse
from app.services.funnel import run_funnel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["evaluation"])

@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_text(request_data: EvaluateRequest, request: Request):
    """
    Evaluate candidate text against D.R.I.V.E. values.
    """
    try:
        response = await run_funnel(request_data.text, request.app.state)
        return response
    except Exception as e:
        logger.exception("Evaluation error: %s", e)
        raise HTTPException(status_code=500, detail=f"Evaluation error: {type(e).__name__}: {e}")

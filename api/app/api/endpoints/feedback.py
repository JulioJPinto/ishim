from fastapi import APIRouter, HTTPException
from app.models.feedback import Feedback
from app.services.feedback_service import record_feedback

router = APIRouter()

@router.post("/")
def submit_feedback(feedback: Feedback):
    try:
        record_feedback(feedback)
        return {"message": "Feedback recorded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel
from typing import List

class FeedbackItem(BaseModel):
    suggested_phrase: str
    rating: int

class Feedback(BaseModel):
    input_phrase: str
    feedback: List[FeedbackItem]

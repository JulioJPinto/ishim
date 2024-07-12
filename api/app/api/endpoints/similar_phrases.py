from fastapi import APIRouter, HTTPException
from app.models.input_phrase import InputPhrase
from app.services.similar_phrases_service import find_top_similar_phrases

router = APIRouter()

@router.post("/")
def get_similar_phrases(input_phrase: InputPhrase):
    try:
        top_similar_phrases = find_top_similar_phrases(input_phrase.phrase)
        return {"input_phrase": input_phrase.phrase, "top_similar_phrases": top_similar_phrases}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

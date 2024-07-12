from fastapi import FastAPI
from app.api.endpoints import feedback, similar_phrases

app = FastAPI()

app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
app.include_router(similar_phrases.router, prefix="/similar_phrases", tags=["similar_phrases"])

# Run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from pydantic import BaseModel

class InputPhrase(BaseModel):
    phrase: str

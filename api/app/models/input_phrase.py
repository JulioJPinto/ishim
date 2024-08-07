from pydantic import BaseModel

class InputPhrase(BaseModel):
    tablename = "" #tbf
    
    id = Column(VARCHAR(255), primary_key=True, index=True) #Phrase ordered alphabetacally
    phrase = Column(VARCHAR(255), index=True) #Original phrase

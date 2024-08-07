import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = f"mssql+pyodbc://{os.getenv('MSSQL_USER')}:{os.getenv('MSSQL_PASSWORD')}@{os.getenv('MSSQL_SERVER')}:{os.getenv('MSSQL_PORT')}/{os.getenv('MSSQL_DATABASE')}?driver=ODBC+Driver+17+for+SQL+Server"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
import os

load_dotenv()

DB_HOST = os.environ["DB_HOST"]
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASS = os.environ["DB_PASS"]

SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"

engine = create_engine(
    url=SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autoflush=False, bind=engine)

Base = declarative_base()

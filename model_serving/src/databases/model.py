from src.databases.mysqldb import Base
from sqlalchemy import Column, Integer, String


class Users(Base):
    __tablename__ = "tblIoTusers"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), nullable=False)
    email = Column(String(250), unique=True)
    hashed_password = Column(String(500))
    role = Column(String(50), nullable=False, default="User")
    scope = Column(String(50), nullable=False, default="Global")
    status = Column(String(50), nullable=False, default="Active")

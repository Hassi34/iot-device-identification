from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from starlette import status
from src.databases.mysqldb import SessionLocal
from src.databases.model import Users
import src.schemas.schema as SCHEMAS
from datetime import datetime, timedelta
from jose import jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordRequestForm
from typing import Annotated
from dotenv import load_dotenv
import os

load_dotenv()

SECRET_KEY = os.environ["JWT_AUTH_SECRET_KEY"]
ALGORITHM = os.environ["JWT_AUTH_ALGORITHM"]

router = APIRouter(prefix="/auth", tags=["Authentication"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]

bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@router.post("/token", response_model=SCHEMAS.Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: db_dependency
):
    user = authenticate_user(
        form_data.username, form_data.password, db  # this will corespond to use email
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate user"
        )
    token = create_access_token(
        user.email, user.role, user.scope, user.status, user.id, timedelta(minutes=5)
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "media_type": "application/json",
    }


def authenticate_user(email: str, password: str, db):
    user = db.query(Users).filter(Users.email == email).first()
    if not user:
        return False
    if not bcrypt_context.verify(password, user.hashed_password):
        return False
    return user


def create_access_token(
    email: str,
    role: str,
    scope: str,
    status: str,
    user_id: int,
    expires_delta: timedelta,
):
    encode = {
        "sub": email,
        "role": role,
        "scope": scope,
        "status": status,
        "id": user_id,
    }
    expires = datetime.utcnow() + expires_delta
    encode.update({"exp": expires})
    return jwt.encode(encode, SECRET_KEY, algorithm=ALGORITHM)

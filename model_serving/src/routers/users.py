from fastapi import APIRouter, Depends, HTTPException
from starlette import status
from src.databases.model import Users
import src.schemas.schema as SCHEMAS
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from typing import Annotated
from src.databases.mysqldb import Base, engine
from dotenv import load_dotenv
import os
from src.routers.authentication import db_dependency
from sqlalchemy.orm import Session

load_dotenv()

SECRET_KEY = os.environ["JWT_AUTH_SECRET_KEY"]
ALGORITHM = os.environ["JWT_AUTH_ALGORITHM"]

router = APIRouter(prefix="/users", tags=["User Management"])

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# db_dependency = Annotated[Session, Depends(get_db)]

bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_bearer = OAuth2PasswordBearer(tokenUrl="auth/token")


async def get_current_user(token: Annotated[str, Depends(oauth2_bearer)]):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        user_id: int = payload.get("id")
        if email is None or user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate user",
            )
        return {"email": email, "id": user_id}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate user"
        )


async def get_current_admin(token: Annotated[str, Depends(oauth2_bearer)]):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        user_id: int = payload.get("id")
        role: str = payload.get("role")
        if email is None or user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate user",
            )
        if role and role.lower() == "admin":
            return {"email": email, "id": user_id, "role": role}
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authorized with the current role assigned",
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate user"
        )


user_dependency = Annotated[dict, Depends(get_current_user)]
admin_dependency = Annotated[dict, Depends(get_current_admin)]


@router.post("/create", status_code=status.HTTP_201_CREATED)
async def create_user(
    db: db_dependency,
    admin: admin_dependency,
    create_user_request: SCHEMAS.CreateUserRequest,
):
    #Base.metadata.create_all(bind=engine)
    if admin:
        user = db.query(Users).filter(Users.email == create_user_request.email).first()
        if not user:
            create_user_model = Users(
                username=create_user_request.username,
                email=create_user_request.email,
                hashed_password=bcrypt_context.hash(create_user_request.password),
                role=create_user_request.role,
                scope=create_user_request.scope,
                status=create_user_request.status,
            )
            db.add(create_user_model)
            db.commit()
            return {
                "success_msg": f"""A new user has been created having the credentials => username = {create_user_request.username}, email = {create_user_request.email}, role = {create_user_request.role}, scope = {create_user_request.scope}, status = {create_user_request.status}"""
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"user with email/username : {create_user_request.email} already exists",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized with the current role assigned",
        )


@router.get("/all", status_code=status.HTTP_200_OK)
def get_all(db: db_dependency, admin: admin_dependency):
    if admin:
        users = db.query(Users).all()
        return users
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized with the current role assigned",
        )


@router.get(
    "/get/{user_id}",
    status_code=status.HTTP_302_FOUND,
    response_model=SCHEMAS.GetUserRequestResponse,
)
async def get_user(db: db_dependency, admin: admin_dependency, user_id: int):
    if admin:
        user = db.query(Users).filter(Users.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "scope": user.scope,
            "status": user.status,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized with the current role assigned",
        )


@router.put(
    "/update/{user_id}",
    response_model=SCHEMAS.UpdateUserRequestResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def update_user(
    user_update_data: SCHEMAS.UpdateUserRequest,
    db: db_dependency,
    admin: admin_dependency,
    user_id: int,
):
    if admin:
        if int(dict(user_update_data)["user_id"]) != int(user_id):
            raise HTTPException(
                status_code=404, detail="Request and the system id does not match"
            )
        db_user = db.query(Users).filter(Users.id == user_id).first()
        if not db_user:
            raise HTTPException(
                status_code=404, detail=f"User with the id = {user_id} not found"
            )
        user_update_data_dict = dict(user_update_data)
        if user_update_data.password:
            user_update_data_dict["hashed_password"] = bcrypt_context.hash(
                user_update_data.password
            )
        for var, value in (user_update_data_dict).items():
            setattr(db_user, var, value) if value else None
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return {
            "user_id": db_user.id,
            "username": db_user.username,
            "email": db_user.email,
            "role": db_user.role,
            "scope": db_user.scope,
            "status": db_user.status,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized with the current role assigned",
        )


@router.delete("/delete/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def destroy(db: db_dependency, admin: admin_dependency, user_id: int):
    if admin:
        db_user = db.query(Users).filter(Users.id == user_id).first()
        if not db_user:
            raise HTTPException(
                status_code=404, detail=f"User with the id = {user_id} not found"
            )
        db.query(Users).filter(Users.id == user_id).delete(synchronize_session=False)
        db.commit()
        return f"User with id = {user_id} has been deleted"
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized with the current role assigned",
        )

from pydantic import BaseModel, Field, EmailStr
from typing import List
from src.configs.pydantic_configs import requested_data_example
from typing import Optional


class CreateUserRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str
    scope: str | None
    status: str | None

    class Config:
        schema_extra = {
            "example": {
                "username": "Jhon Doe",
                "email": "user@example.com",
                "password": "SuperSecuredPassword",
                "role": "User",
                "scope": "Global",
                "status": "Active",
            }
        }


class GetUserRequestResponse(BaseModel):
    user_id: str
    username: str
    email: EmailStr
    role: str
    scope: str
    status: str


class UpdateUserRequestResponse(BaseModel):
    user_id: str
    username: str | None
    email: EmailStr
    role: str | None
    scope: str | None
    status: str | None


class UpdateUserRequest(BaseModel):
    user_id: str
    username: str | None
    password: str | None
    email: EmailStr
    role: str | None
    scope: str | None
    status: str | None


class Token(BaseModel):
    access_token: str
    token_type: str
    media_type: Optional[str]


class ShowResults(BaseModel):
    predicted_class_label: str = None
    class_indices: dict = None
    prediction_probabilities: List[float] = None


# class Login(BaseModel):
#     username: str
#     password : str
class Prediction(BaseModel):
    request_data: dict | None = Field(default=None)
    request_data = requested_data_example

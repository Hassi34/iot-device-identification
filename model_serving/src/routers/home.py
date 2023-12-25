from fastapi import APIRouter
from starlette.responses import RedirectResponse
from src.databases.mysqldb import Base, engine

router = APIRouter(tags=["Home"])


@router.get("/")
async def index():
    Base.metadata.create_all(bind=engine)
    return RedirectResponse(url="/docs")

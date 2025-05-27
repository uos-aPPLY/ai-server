from fastapi import APIRouter
from fastapi.responses import JSONResponse


router = APIRouter()

@router.get("/")
async def check():
    return JSONResponse(content={"status": "ok", "message": "AI Server is running."})

@router.get("/healthcheck")
async def healthcheck():
    return JSONResponse(content={"status": "ok", "message": "AI Server is running."})
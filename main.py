from fastapi import FastAPI
from planogram_route import router as planogram_router

app = FastAPI()

app.include_router(planogram_router, prefix="/planogram", tags=["Planogram"])
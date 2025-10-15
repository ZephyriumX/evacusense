from fastapi import FastAPI
from .api import router as api_router

app = FastAPI(title="EvacuSense - API")
app.include_router(api_router)

@app.get("/")
def root():
    return {"status": "evacusense backend running"}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api import endpoints

app = FastAPI(
    title="Data Leakage Detection API",
    description="Automated Data Leakage Detection for ML Pipelines",
    version="1.0.0"
)

# CORS Configuration
origins = [
    "http://localhost:3000", # Next.js frontend
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(endpoints.router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Data Leakage Detection API is running. Visit /docs for Swagger UI."}

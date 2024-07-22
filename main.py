import api
from fastapi import FastAPI, Depends
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

from fastapi.middleware.cors import CORSMiddleware

# User can send only 1 request per 3 seconds
app = FastAPI()

app.include_router(api.router)

# Allow requests from all origins (You can specify specific origins if needed)
origins = ["*"]

# Add CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import api
from fastapi import FastAPI, Depends
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from redis import Redis
from rq import Queue
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    global queue, active_connections, redis_conn
    # Creating Redis instance at localhost with port 6479
    redis_conn = Redis()
    await FastAPILimiter.init(redis=redis_conn)
    queue = Queue(connection=redis_conn)
    active_connections = {}
    yield
    # close all the active WebSocket connections
    for job_id, websocket in active_connections.items():
        if websocket is not None:
            await websocket.close()
    active_connections.clear()

# User can send only 1 request per 3 seconds
app = FastAPI(dependencies=[Depends(RateLimiter(times=1, seconds=3))], lifespan=lifespan)


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

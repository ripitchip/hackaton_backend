from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import file, users

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)   
app.include_router(users.router)
app.include_router(file.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}

import os
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import gemini_utils as ai

app = FastAPI(title="PocketSmart AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = os.path.dirname(os.path.abspath(__file__))


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE, "templates", "index.html"))


@app.post("/generate-home")
async def generate_home(budget: float = Form(...), room: str = Form(...), items: str = Form(...)):
    return ai.home_plan(budget, room, items)


@app.post("/generate-party")
async def generate_party(budget: float = Form(...), guests: int = Form(...),
                         event: str = Form(...), venue: str = Form("")):
    return ai.party_plan(budget, guests, event, venue)


@app.post("/generate-jewelry")
async def generate_jewelry(budget: float = Form(...), occasion: str = Form(...),
                           style: str = Form(""), image: Optional[UploadFile] = File(None)):
    data, mime = None, None
    if image and image.filename:
        data = await image.read()
        mime = image.content_type
    return ai.jewelry_plan(budget, occasion, style, data, mime)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
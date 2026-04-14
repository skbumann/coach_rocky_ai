import json
import asyncio
from typing import Optional
from openai import OpenAI
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from src.rag_helper import *
from contextlib import asynccontextmanager
import logging
import time
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress some logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# FastAPI app
app = FastAPI(title="Personal Coach AI", version="1.0.0")

@asynccontextmanager
async def timer(name):
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{name}: {elapsed:.2f}s")

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the chat HTML interface"""
    with open("templates/chat.html", "r") as f:
        return f.read()

@app.post("/get")
async def chat(msg: str = Form(...)):
    """
    Handle chat messages and return RAG responses
    
    Args:
        msg: User message from form submission
        
    Returns:
        Generated response from RAG chain
    """
    async with timer("Total request"):
        if not msg or msg.strip() == "":
            raise HTTPException(status_code=400, detail="No input received.")

        try:
            async with timer("RAG chain"):
                response = run_rag_agent(user_prompt=msg)
            logger.info(f"Response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error: {e}")
            raise HTTPException(status_code=500, detail="There was an error processing your request.")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)

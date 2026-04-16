import json
import asyncio
import uvicorn
from typing import Optional
from openai import OpenAI
from fastapi import FastAPI, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from requests_oauthlib import OAuth2Session
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

# OAuth Config 
client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
redirect_uri = "https://localhost:8000/callback" 
auth_base_url = "https://www.strava.com/oauth/authorize"
token_url = "https://www.strava.com/api/v3/oauth/token"

session = OAuth2Session(client_id=client_id, redirect_uri=redirect_uri)
session.scope = ["activity:read_all"]

@asynccontextmanager
async def timer(name):
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{name}: {elapsed:.2f}s")

@app.get("/")#, response_class=HTMLResponse)
def root():
    authorization_url, _ = session.authorization_url(auth_base_url)
    return RedirectResponse(authorization_url)

@app.get("/callback")
#async def index():
#    """Serve the chat HTML interface"""
#    with open("templates/chat.html", "r") as f:
#        return f.read()
async def callback(request: Request, background_tasks: BackgroundTasks):
    # Capture the full HTTPS URL
    authorization_response = str(request.url)
    
    session_user = OAuth2Session(client_id=client_id, redirect_uri=redirect_uri)
    session_user.scope = ["activity:read_all"]
    session_user.fetch_token(
        token_url=token_url,
        client_id=client_id,
	    client_secret=client_secret,
        authorization_response=authorization_response,
        include_client_id=True
    )

    # Get individual activity (access to activity description)
    #response = session_user.get("https://www.strava.com/api/v3/activities/17967927452") 

    # Get list of all activities (doesn't show activity description)
    params = {'per_page': 200, 'page': 1}
    response = session_user.get("https://www.strava.com/api/v3/athlete/activities/", params=params) 

    # Extract the JSON data from the response
    activities_data = response.json()

    with open('my_strava_data.json', 'w') as f:
        json.dump(activities_data, f, indent=4)

    #load_data()
    #data = response.text
    #print(len(activities_data))
    #return None #activities_data #{"status": "Authenticated", "token": token}
    background_tasks.add_task(load_data)
    
    # 4. Serve the Chat Interface
    with open("templates/chat.html", "r") as f:
        return HTMLResponse(content=f.read())

    #with open("templates/chat.html", "r") as f:
    #    html_content = f.read()
    
    #return HTMLResponse(content=html_content, status_code=200)

#async def index():
#    """Serve the chat HTML interface"""
#    with open("templates/chat.html", "r") as f:
#        return f.read()


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
    uvicorn.run(
    "main:app", 
    host="127.0.0.1", 
    port=8000, 
    reload=True,
    ssl_keyfile="./localhost-key.pem", 
    ssl_certfile="./localhost.pem"
)
    
    #port = int(os.environ.get("PORT", 5000))
    #uvicorn.run(app, host="0.0.0.0", port=port)

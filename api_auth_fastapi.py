import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# OAuth Config (Ensure your Strava Callback is set to https://localhost:8000/callback)
client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
redirect_uri = "https://localhost:8000/callback" 
auth_base_url = "https://www.strava.com/oauth/authorize"
token_url = "https://www.strava.com/api/v3/oauth/token"

session = OAuth2Session(client_id=client_id, redirect_uri=redirect_uri)
session.scope = ["activity:read_all"]

@app.get("/")
def root():
    authorization_url, _ = session.authorization_url(auth_base_url)
    return RedirectResponse(authorization_url)
    #return {"login_url": authorization_url}

@app.get("/callback")
async def callback(request: Request):
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
    response = session_user.get("https://www.strava.com/api/v3/athlete/activities") 
    data = response.text
    print(data)
    return data #{"status": "Authenticated", "token": token}

if __name__ == "__main__":
    # Point uvicorn to your mkcert files
    uvicorn.run(
        "main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        ssl_keyfile="./localhost-key.pem", 
        ssl_certfile="./localhost.pem"
    )

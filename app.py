import json
import asyncio
import uvicorn
from typing import Optional
from openai import OpenAI
from fastapi import FastAPI, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from requests_oauthlib import OAuth2Session
from src.rag_helper import (
    get_schema_name,
    provision_tenant_schema,
    load_data_for_user,
    run_rag_agent,
)
from contextlib import asynccontextmanager
import logging
import time
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Personal Coach AI", version="2.0.0")

# Session middleware — SESSION_SECRET_KEY must be a strong random value in prod.
# Generate one with: python -c "import secrets; print(secrets.token_hex(32))"
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "change-me-in-production"),
    session_cookie="strava_session",
    max_age=86400,   # 1 day
    https_only=True, # Set False only for local HTTP dev
)

# ── OAuth Config ───────────────────────────────────────────────────────────────
CLIENT_ID     = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI  = os.getenv("REDIRECT_URI", "https://localhost:8000/callback")
AUTH_BASE_URL = "https://www.strava.com/oauth/authorize"
TOKEN_URL     = "https://www.strava.com/api/v3/oauth/token"

# ── Helpers ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def timer(name: str):
    start = time.time()
    yield
    logger.info(f"{name}: {time.time() - start:.2f}s")


def _require_athlete(request: Request) -> str:
    """Return athlete_id from session or raise 401."""
    athlete_id = request.session.get("athlete_id")
    if not athlete_id:
        raise HTTPException(status_code=401, detail="Not authenticated. Please visit / to log in.")
    return athlete_id


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Redirect to Strava OAuth consent screen."""
    session = OAuth2Session(client_id=CLIENT_ID, redirect_uri=REDIRECT_URI)
    session.scope = ["activity:read_all"]
    authorization_url, _ = session.authorization_url(AUTH_BASE_URL)
    return RedirectResponse(authorization_url)


@app.get("/callback")
async def callback(request: Request, background_tasks: BackgroundTasks):
    """
    1. Exchange Strava auth code for tokens.
    2. Fetch the athlete's profile to get their unique ID.
    3. Provision a dedicated PostgreSQL schema for this tenant (idempotent).
    4. Store athlete_id + tokens in the session cookie.
    5. Kick off background activity ingestion.
    6. Serve the chat UI.
    """
    authorization_response = str(request.url)

    session_user = OAuth2Session(client_id=CLIENT_ID, redirect_uri=REDIRECT_URI)
    session_user.scope = ["activity:read_all"]
    token = session_user.fetch_token(
        token_url=TOKEN_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        authorization_response=authorization_response,
        include_client_id=True,
    )

    # ── Identify the athlete ───────────────────────────────────────────────────
    athlete_resp = session_user.get("https://www.strava.com/api/v3/athlete")
    athlete_resp.raise_for_status()
    athlete_data = athlete_resp.json()
    athlete_id   = str(athlete_data["id"])

    # ── Persist tenant schema (no-op if already exists) ───────────────────────
    schema = get_schema_name(athlete_id)
    provision_tenant_schema(athlete_id)

    # ── Store session ──────────────────────────────────────────────────────────
    request.session["athlete_id"]    = athlete_id
    request.session["access_token"]  = token["access_token"]
    request.session["refresh_token"] = token.get("refresh_token", "")

    # ── Fetch activities (up to 200) and ingest in background ─────────────────
    params           = {"per_page": 200, "page": 1}
    activities_resp  = session_user.get(
        "https://www.strava.com/api/v3/athlete/activities/", params=params
    )
    activities_resp.raise_for_status()
    activities_data  = activities_resp.json()

    background_tasks.add_task(load_data_for_user, athlete_id, activities_data, schema)
    logger.info(f"Queued ingestion of {len(activities_data)} activities for athlete {athlete_id}")

    # ── Serve chat UI ──────────────────────────────────────────────────────────
    with open("templates/chat.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/get")
async def chat(request: Request, msg: str = Form(...)):
    """
    Handle a chat message from the UI.
    The tenant schema is derived from the session, so each user only ever
    queries their own data.
    """
    athlete_id = _require_athlete(request)

    async with timer("Total request"):
        if not msg or not msg.strip():
            raise HTTPException(status_code=400, detail="No input received.")

        schema = get_schema_name(athlete_id)

        try:
            async with timer("RAG agent"):
                response = run_rag_agent(user_prompt=msg, schema=schema)
            logger.info(f"[athlete={athlete_id}] Response: {response}")
            return response
        except Exception as e:
            logger.error(f"[athlete={athlete_id}] Error: {e}")
            raise HTTPException(status_code=500, detail="Error processing your request.")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/me")
async def me(request: Request):
    """Return basic session info (useful for debugging)."""
    athlete_id = _require_athlete(request)
    return {"athlete_id": athlete_id, "schema": get_schema_name(athlete_id)}


# ── Entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Local dev only — in production, Fargate/ALB handles SSL termination.
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        ssl_keyfile="./localhost-key.pem",
        ssl_certfile="./localhost.pem",
    )

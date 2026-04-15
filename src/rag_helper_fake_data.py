import os
from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env' 
load_dotenv(dotenv_path=env_path)
import json
import asyncio
from typing import Optional
import psycopg2
import psycopg2.extras
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import logging
import time
from src.prompt import system_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress some logs

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Configure

DB_URL = os.environ.get('DB_URL')
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
VECTOR_DIMS = 1536

openai_api_key= os.environ.get('OPENAI_API_KEY')

# DB helpers

def get_conn():
    #dsn = DB_URL
    dsn = os.getenv("DATABASE_URL")
    return psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)


# Ingest the JSON data into PostgreSQL

def ingest_activity(activity: dict) -> None:
    """Insert a raw activity JSON into PostgreSQL, parsed into typed columns."""

    sql_activity = """
        INSERT INTO activities (
            id, upload_id, external_id, athlete_id,
            name, description,
            activity_type, sport_type,
            distance_meters, moving_time_seconds, elapsed_time_seconds,
            total_elevation_gain, average_speed, max_speed,
            average_heartrate, max_heartrate, calories,
            pr_count, achievement_count, kudos_count,
            start_date, start_date_local, timezone,
            gear_id, trainer, commute, private,
            raw_json
        )
        VALUES (
            %(id)s, %(upload_id)s, %(external_id)s, %(athlete_id)s,
            %(name)s, %(description)s,
            %(activity_type)s, %(sport_type)s,
            %(distance_meters)s, %(moving_time_seconds)s, %(elapsed_time_seconds)s,
            %(total_elevation_gain)s, %(average_speed)s, %(max_speed)s,
            %(average_heartrate)s, %(max_heartrate)s, %(calories)s,
            %(pr_count)s, %(achievement_count)s, %(kudos_count)s,
            %(start_date)s, %(start_date_local)s, %(timezone)s,
            %(gear_id)s, %(trainer)s, %(commute)s, %(private)s,
            %(raw_json)s
        )
        ON CONFLICT (id) DO NOTHING
    """

    params = {
        "id":                   activity["id"],
        "upload_id":            activity.get("upload_id"),
        "external_id":          activity.get("external_id"),
        "athlete_id":           activity["athlete"]["id"],
        # unstructured text
        "name":                 activity.get("name"),
        "description":          activity.get("description"),
        # structured fields
        "activity_type":        activity.get("type"),
        "sport_type":           activity.get("sport_type"),
        "distance_meters":      activity.get("distance"),
        "moving_time_seconds":  activity.get("moving_time"),
        "elapsed_time_seconds": activity.get("elapsed_time"),
        "total_elevation_gain": activity.get("total_elevation_gain"),
        "average_speed":        activity.get("average_speed"),
        "max_speed":            activity.get("max_speed"),
        "average_heartrate":    activity.get("average_heartrate"),
        "max_heartrate":        activity.get("max_heartrate"),
        "calories":             activity.get("calories"),
        "pr_count":             activity.get("pr_count", 0),
        "achievement_count":    activity.get("achievement_count", 0),
        "kudos_count":          activity.get("kudos_count", 0),
        "start_date":           activity.get("start_date"),
        "start_date_local":     activity.get("start_date_local"),
        "timezone":             activity.get("timezone"),
        "gear_id":              activity.get("gear_id"),
        "trainer":              activity.get("trainer", False),
        "commute":              activity.get("commute", False),
        "private":              activity.get("private", False),
        "raw_json":             json.dumps(activity),
    }

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_activity, params)
        conn.commit()

    print(f"Ingested activity {activity['id']}: '{activity.get('name')}'")


def embed_text(text: str) -> list[float]:
    """Call OpenAI embeddings API and return the vector."""
    client = OpenAI(api_key=openai_api_key) 
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def build_chunk_text(activity: dict, chunk_type: str = "combined") -> str:
    """
    Construct the text string that will be embedded.
    'combined' merges name + description for richer semantic signal.
    """
    name = activity.get("name", "")
    desc = activity.get("description", "")

    if chunk_type == "name":
        return name
    elif chunk_type == "description":
        return desc
    else:  # combined 
        parts = []
        if name:
            parts.append(name)
        if desc:
            parts.append(desc)
        return ". ".join(parts)


def embed_activity(activity_id: int, chunk_text: str, chunk_type: str = "combined") -> None:
    """Generate and store embedding for an activity's text chunk."""

    vector = embed_text(chunk_text)

    sql = """
        INSERT INTO activity_embeddings (activity_id, chunk_type, chunk_text, embedding, embedding_model)
        VALUES (%s, %s, %s, %s::vector, %s)
        ON CONFLICT (activity_id, chunk_text) DO NOTHING;
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (activity_id, chunk_type, chunk_text, str(vector), EMBEDDING_MODEL))
        conn.commit()

    print(f"Embedded activity {activity_id} [{chunk_type}]")


def ingest_and_embed(activity: dict) -> None:
    """Full pipeline: ingest JSON → store → embed."""
    ingest_activity(activity)

    chunk_text = build_chunk_text(activity, chunk_type="combined")
    if chunk_text.strip():
        embed_activity(activity["id"], chunk_text, chunk_type="combined")


# Semantic retrieval for the unstructured data (i.e. captions)

def retrieve_similar_activities(
    query: str,
    top_k: int = 5,
    activity_type: Optional[str] = None,
    min_distance_meters: Optional[float] = None,
    since_date: Optional[str] = None,
) -> list[dict]:
    """
    Embed the query, then find the most semantically similar activities.
    Optionally pre-filter by structured metadata (type, distance, date).
    """

    query_vector = embed_text(query)

    # Build dynamic WHERE clause for metadata pre-filtering
    filters = ["e.chunk_type = 'combined'"]
    params: list = []

    if activity_type:
        filters.append("a.activity_type = %s")
        params.append(activity_type)

    if min_distance_meters:
        filters.append("a.distance_meters >= %s")
        params.append(min_distance_meters)

    if since_date:
        filters.append("a.start_date >= %s")
        params.append(since_date)

    where_clause = " AND ".join(filters)

    sql = f"""
        SELECT
            a.id,
            a.name,
            a.description,
            a.activity_type,
            a.distance_meters,
            a.moving_time_seconds,
            a.average_heartrate,
            a.calories,
            a.start_date,
            e.chunk_text,
            1 - (e.embedding <=> %s::vector) AS similarity
        FROM activity_embeddings e
        JOIN activities a ON a.id = e.activity_id
        WHERE {where_clause}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """

    vector_str = str(query_vector)
    all_params = [vector_str] + params + [vector_str, top_k]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, all_params)
            return cur.fetchall()
        
# Define tools for routing

# 1. Structured stats analysis
@tool
def get_strava_stats(sql_query: str):
    """Execute a SQL query to get structured stats. Use for: averages, distance totals, heart rate, or date filters."""
    print(">>> get_strava_stats called")
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query)
                return json.dumps(cur.fetchall(), default=str)
    except Exception as e:
        return f"Error executing SQL: {e}"

# 2. Semantic search
@tool
def get_activity_vibes(semantic_query: str):
    """Search for activities based on feelings, mood, or descriptive themes. Use for: 'When was I happy?', 'sore legs', 'motivation'."""
    print(">>> get_activity_vibes called")
    results = retrieve_similar_activities(query=semantic_query, top_k=3)
    return json.dumps(results, default=str)

# 3. Get more holistic trends of the athlete
@tool
def get_training_baseline(weeks: int = 64):
    """
    Get weekly mileage, run frequency, and longest run for the last N weeks.
    Always call this first when answering questions about training goals or safety.
    """
    print(">>> get_training_baseline called with weeks =", weeks)
    sql = f"""
        WITH weekly AS (
            SELECT
                DATE_TRUNC('week', start_date)        AS week,
                COUNT(*)                               AS runs_that_week,
                SUM(distance_meters) / 1000.0          AS weekly_km,
                SUM(distance_meters) / 1609.34         AS weekly_miles,
                MAX(distance_meters) / 1609.34         AS longest_run_miles,
                AVG(average_heartrate)                 AS avg_hr
            FROM activities
            WHERE LOWER(TRIM(activity_type)) = 'run'
              AND start_date >= NOW() - INTERVAL '{weeks} weeks'
            GROUP BY week
            ORDER BY week DESC
        )
        SELECT
            week,
            runs_that_week,
            ROUND(weekly_miles::numeric, 2)        AS weekly_miles,
            ROUND(longest_run_miles::numeric, 2)   AS longest_run_miles,
            ROUND(avg_hr::numeric, 1)              AS avg_hr
        FROM weekly;
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()

        if not rows:
            return "No running data found for this period."

        total_miles = sum(r["weekly_miles"] for r in rows)
        avg_weekly = total_miles / len(rows)
        avg_frequency = sum(r["runs_that_week"] for r in rows) / len(rows)
        longest = max(r["longest_run_miles"] for r in rows)

        summary = {
            "weeks_analyzed": len(rows),
            "avg_weekly_miles": round(avg_weekly, 2),
            "avg_runs_per_week": round(avg_frequency, 1),
            "longest_recent_run_miles": round(longest, 2),
            "total_miles_in_period": round(total_miles, 2),
            "weekly_breakdown": list(rows)
        }
        return json.dumps(summary, default=str)
    except Exception as e:
        logger.error(f"get_training_baseline failed: {e}")
        return f"Error fetching baseline: {e}"

# Initialize model and pipe into LangGraph's create_react_agent
_agent_cache = None

def get_agent():
    global _agent_cache
    if _agent_cache is None:
        print("Initializing AI Agent for the first time...")
        tools = [get_strava_stats, get_activity_vibes, get_training_baseline]
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
        _agent_cache = create_react_agent(model, tools)
    return _agent_cache

# The main RAG workhorse
def run_rag_agent(user_prompt: str):
    agent = get_agent()  
    response = agent.invoke({
        "messages": [
            ("system", system_prompt),
            ("user", user_prompt)
        ]
    })
    return response["messages"][-1].content

# Load JSON data into DB
def load_data():
    # Path to your JSON file
    json_file_path = "data/strava_activities.json"

    try:
        # 1. Load the full list of activities
        with open(json_file_path, 'r') as f:
            activities_list = json.load(f)
        
        print(f"Found {len(activities_list)} activities. Starting ingestion...")

        # 2. Loop through and process each one
        for activity in activities_list:
            try:
                ingest_and_embed(activity)
            except Exception as e:
                print(f"Failed to process activity {activity.get('id')}: {e}")

        print("\nAll activities processed successfully!")

    except FileNotFoundError:
        print(f"Error: Could not find '{json_file_path}'. Check the filename.")
    except json.JSONDecodeError:
        print(f"Error: '{json_file_path}' is not a valid JSON file.")

if __name__ == "__main__":
    load_data()
"""
rag_helper.py — Multi-tenant RAG pipeline
==========================================
Each Strava user gets an isolated PostgreSQL schema: athlete_<id>.
All DB helpers accept a `schema` parameter so no user ever touches
another user's data.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

import json
import re
from typing import Optional

import psycopg2
import psycopg2.extras
from psycopg2 import sql as pgsql  # parameterised identifiers — avoids SQL injection
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import logging

from src.prompt import system_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ── Config ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o-mini"
VECTOR_DIMS     = 1536

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY")

# Allowlist of columns the LLM is permitted to query.
# Prevents prompt-injection attacks via crafted SQL in get_strava_stats.
ALLOWED_COLUMNS = {
    "id", "athlete_id", "name", "sport_type", "distance_meters",
    "moving_time_seconds", "elapsed_time_seconds", "total_elevation_gain",
    "average_speed", "max_speed", "average_heartrate", "max_heartrate",
    "average_watts", "kilojoules", "comment_count", "pr_count",
    "achievement_count", "kudos_count", "athlete_count",
    "start_lat", "start_long", "end_lat", "end_long",
    "elev_high", "elev_low", "start_date", "start_date_local", "timezone",
    "gear_id", "trainer", "commute", "private",
}

# ── Tenant helpers ─────────────────────────────────────────────────────────────

def get_schema_name(athlete_id: str) -> str:
    """Return the PostgreSQL schema name for a given athlete."""
    # Sanitise: only digits allowed in athlete IDs from Strava
    safe_id = re.sub(r"\D", "", str(athlete_id))
    return f"athlete_{safe_id}"


def provision_tenant_schema(athlete_id: str) -> None:
    """
    Idempotently create the schema and tables for a new tenant.
    Safe to call on every login — IF NOT EXISTS guards handle repeats.
    """
    schema = get_schema_name(athlete_id)
    logger.info(f"Provisioning schema: {schema}")

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Schema
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

            # Core activities table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema}.activities (
                    id                      BIGINT PRIMARY KEY,
                    athlete_id              BIGINT,
                    name                    TEXT,
                    sport_type              TEXT,
                    distance_meters         FLOAT,
                    moving_time_seconds     INT,
                    elapsed_time_seconds    INT,
                    total_elevation_gain    FLOAT,
                    average_speed           FLOAT,
                    max_speed               FLOAT,
                    average_heartrate       FLOAT,
                    max_heartrate           FLOAT,
                    average_watts           FLOAT,
                    kilojoules              FLOAT,
                    comment_count           INT DEFAULT 0,
                    pr_count                INT DEFAULT 0,
                    achievement_count       INT DEFAULT 0,
                    kudos_count             INT DEFAULT 0,
                    athlete_count           INT DEFAULT 0,
                    start_lat               FLOAT,
                    start_long              FLOAT,
                    end_lat                 FLOAT,
                    end_long                FLOAT,
                    elev_high               FLOAT,
                    elev_low                FLOAT,
                    start_date              TIMESTAMPTZ,
                    start_date_local        TIMESTAMPTZ,
                    timezone                TEXT,
                    gear_id                 TEXT,
                    trainer                 BOOLEAN DEFAULT FALSE,
                    commute                 BOOLEAN DEFAULT FALSE,
                    private                 BOOLEAN DEFAULT FALSE,
                    raw_json                JSONB
                )
            """)

            # Embeddings table — requires pgvector extension on the DB
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema}.activity_embeddings (
                    id              SERIAL PRIMARY KEY,
                    activity_id     BIGINT NOT NULL REFERENCES {schema}.activities(id),
                    chunk_type      TEXT,
                    chunk_text      TEXT,
                    embedding       vector({VECTOR_DIMS}),
                    embedding_model TEXT,
                    UNIQUE (activity_id, chunk_text)
                )
            """)

        conn.commit()
    logger.info(f"Schema {schema} is ready.")


# ── DB connection ──────────────────────────────────────────────────────────────

def get_conn():
    dsn = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)
    
    # DEBUG: Print the physical location of the DB
    #with conn.cursor() as cur:
    #    cur.execute("SELECT current_database(), inet_server_addr(), inet_server_port();")
    #    db_info = cur.fetchone()
    #    print(f"🔍 CONNECTED TO: {db_info['current_database']} "
    #          f"ON IP: {db_info['inet_server_addr']} "
    #          f"PORT: {db_info['inet_server_port']}")
    return conn
    #dsn = os.getenv("DATABASE_URL")
    #return psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)


# ── Ingestion ──────────────────────────────────────────────────────────────────

def ingest_activity(activity: dict, schema: str) -> None:
    """Insert a raw activity JSON into the tenant's activities table."""

    sql_activity = f"""
        INSERT INTO {schema}.activities (
            id, athlete_id, name, sport_type,
            distance_meters, moving_time_seconds, elapsed_time_seconds,
            total_elevation_gain, average_speed, max_speed,
            average_heartrate, max_heartrate, average_watts, kilojoules,
            comment_count, pr_count, achievement_count, kudos_count,
            athlete_count, start_lat, start_long, end_lat, end_long,
            elev_high, elev_low, start_date, start_date_local, timezone,
            gear_id, trainer, commute, private, raw_json
        )
        VALUES (
            %(id)s, %(athlete_id)s, %(name)s, %(sport_type)s,
            %(distance_meters)s, %(moving_time_seconds)s, %(elapsed_time_seconds)s,
            %(total_elevation_gain)s, %(average_speed)s, %(max_speed)s,
            %(average_heartrate)s, %(max_heartrate)s, %(average_watts)s, %(kilojoules)s,
            %(comment_count)s, %(pr_count)s, %(achievement_count)s, %(kudos_count)s,
            %(athlete_count)s, %(start_lat)s, %(start_long)s, %(end_lat)s, %(end_long)s,
            %(elev_high)s, %(elev_low)s, %(start_date)s, %(start_date_local)s, %(timezone)s,
            %(gear_id)s, %(trainer)s, %(commute)s, %(private)s, %(raw_json)s
        )
        ON CONFLICT (id) DO NOTHING
    """

    start_coords = activity.get("start_latlng") or []
    end_coords   = activity.get("end_latlng") or []

    params = {
        "id":                   activity["id"],
        "athlete_id":           activity["athlete"]["id"],
        "name":                 activity.get("name"),
        "sport_type":           activity.get("sport_type"),
        "distance_meters":      activity.get("distance"),
        "moving_time_seconds":  activity.get("moving_time"),
        "elapsed_time_seconds": activity.get("elapsed_time"),
        "total_elevation_gain": activity.get("total_elevation_gain"),
        "average_speed":        activity.get("average_speed"),
        "max_speed":            activity.get("max_speed"),
        "average_heartrate":    activity.get("average_heartrate"),
        "max_heartrate":        activity.get("max_heartrate"),
        "average_watts":        activity.get("average_watts"),
        "kilojoules":           activity.get("kilojoules"),
        "comment_count":        activity.get("comment_count", 0),
        "pr_count":             activity.get("pr_count", 0),
        "achievement_count":    activity.get("achievement_count", 0),
        "kudos_count":          activity.get("kudos_count", 0),
        "athlete_count":        activity.get("athlete_count", 0),
        "start_lat":            start_coords[0] if len(start_coords) >= 2 else None,
        "start_long":           start_coords[1] if len(start_coords) >= 2 else None,
        "end_lat":              end_coords[0]   if len(end_coords) >= 2   else None,
        "end_long":             end_coords[1]   if len(end_coords) >= 2   else None,
        "elev_high":            activity.get("elev_high"),
        "elev_low":             activity.get("elev_low"),
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


def embed_text(text: str) -> list[float]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def build_chunk_text(activity: dict, chunk_type: str = "name") -> str:
    name = activity.get("name", "")
    desc = activity.get("description", "")
    if chunk_type == "name":
        return name
    elif chunk_type == "description":
        return desc
    parts = [p for p in [name, desc] if p]
    return ". ".join(parts)


def embed_activity(activity_id: int, chunk_text: str, schema: str, chunk_type: str = "name") -> None:
    vector = embed_text(chunk_text)
    sql = f"""
        INSERT INTO {schema}.activity_embeddings (activity_id, chunk_type, chunk_text, embedding, embedding_model)
        VALUES (%s, %s, %s, %s::vector, %s)
        ON CONFLICT (activity_id, chunk_text) DO NOTHING
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (activity_id, chunk_type, chunk_text, str(vector), EMBEDDING_MODEL))
        conn.commit()


def ingest_and_embed(activity: dict, schema: str) -> None:
    ingest_activity(activity, schema)
    chunk_text = build_chunk_text(activity, chunk_type="name")
    if chunk_text.strip():
        embed_activity(activity["id"], chunk_text, schema, chunk_type="name")


def load_data_for_user(athlete_id: str, activities_data: list, schema: str) -> None:
    """
    Background task: ingest a list of Strava activity dicts into the
    tenant's schema. Called after /callback with already-fetched data.
    """
    logger.info(f"[{schema}] Starting ingestion of {len(activities_data)} activities...")
    success, failed = 0, 0
    for activity in activities_data:
        try:
            ingest_and_embed(activity, schema)
            success += 1
        except Exception as e:
            logger.error(f"[{schema}] Failed to process activity {activity.get('id')}: {e}")
            failed += 1
    logger.info(f"[{schema}] Ingestion complete — {success} succeeded, {failed} failed.")


# ── Semantic retrieval ─────────────────────────────────────────────────────────

def retrieve_similar_activities(
    query: str,
    schema: str,
    top_k: int = 5,
    sport_type: Optional[str] = None,
    min_distance_meters: Optional[float] = None,
    since_date: Optional[str] = None,
) -> list[dict]:
    query_vector = embed_text(query)

    filters = ["e.chunk_type = 'name'"]
    params: list = []

    if sport_type:
        filters.append("a.sport_type = %s")
        params.append(sport_type)
    if min_distance_meters:
        filters.append("a.distance_meters >= %s")
        params.append(min_distance_meters)
    if since_date:
        filters.append("a.start_date >= %s")
        params.append(since_date)

    where_clause = " AND ".join(filters)
    vector_str   = str(query_vector)

    sql = f"""
        SELECT
            a.id, a.name, a.sport_type, a.distance_meters,
            a.moving_time_seconds, a.average_heartrate, a.start_date,
            e.chunk_text,
            1 - (e.embedding <=> %s::vector) AS similarity
        FROM {schema}.activity_embeddings e
        JOIN {schema}.activities a ON a.id = e.activity_id
        WHERE {where_clause}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """

    all_params = [vector_str] + params + [vector_str, top_k]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, all_params)
            return cur.fetchall()


# ── SQL safety guard ───────────────────────────────────────────────────────────

def _scope_and_validate_sql(raw_sql: str, schema: str) -> str:
    """
    1. Rewrite unqualified table references to schema-qualified ones.
    2. Block any statement that isn't a SELECT.
    3. Ensure the query only references the caller's schema.

    This is a best-effort defence; keep the LLM system prompt tight too.
    """
    normalised = raw_sql.strip().upper()
    if not normalised.startswith("SELECT"):
        raise ValueError("Only SELECT statements are permitted.")

    # Rewrite bare table names → schema-qualified
    scoped = raw_sql.replace("FROM activities", f"FROM {schema}.activities")
    scoped = scoped.replace("from activities", f"from {schema}.activities")

    # Paranoia check: no other schema must appear in the query
    other_schema_pattern = re.compile(r"athlete_\d+\.", re.IGNORECASE)
    for match in other_schema_pattern.finditer(scoped):
        if not scoped[match.start():].startswith(schema):
            raise ValueError("Cross-tenant schema reference detected — query rejected.")

    return scoped


# ── LangGraph tools (schema-bound factory) ────────────────────────────────────

def create_tools_for_schema(schema: str):
    """
    Return a fresh set of LangGraph tools bound to a specific tenant schema.
    Tools are cheap to create; do NOT cache them globally since the schema
    differs per user.
    """

    @tool
    def get_strava_stats(sql_query: str) -> str:
        """
        Execute a read-only SQL query against the athlete's activities table.
        Use for: totals, averages, distance, heart rate, counts, date filters.
        The table name is simply 'activities' — do NOT add a schema prefix.
        Only SELECT statements are allowed.
        """
        logger.info(f"[{schema}] get_strava_stats called")
        try:
            safe_sql = _scope_and_validate_sql(sql_query, schema)
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(safe_sql)
                    return json.dumps(cur.fetchall(), default=str)
        except ValueError as ve:
            return f"Query rejected: {ve}"
        except Exception as e:
            return f"Error executing SQL: {e}"

    @tool
    def get_activity_vibes(semantic_query: str) -> str:
        """
        Semantic search over activity names/descriptions.
        Use for feelings, mood, or qualitative themes:
        'When did I feel strong?', 'sore legs', 'evening run', 'motivation'.
        """
        logger.info(f"[{schema}] get_activity_vibes called")
        results = retrieve_similar_activities(query=semantic_query, schema=schema, top_k=3)
        return json.dumps(results, default=str)

    @tool
    def get_training_baseline(weeks: int = 64) -> str:
        """
        Weekly mileage, run frequency, and longest run for the last N weeks.
        Call this first when answering training goal or safety questions.
        """
        logger.info(f"[{schema}] get_training_baseline called with weeks={weeks}")
        sql = f"""
            WITH weekly AS (
                SELECT
                    DATE_TRUNC('week', start_date)        AS week,
                    COUNT(*)                              AS runs_that_week,
                    SUM(distance_meters) / 1609.34        AS weekly_miles,
                    MAX(distance_meters) / 1609.34        AS longest_run_miles,
                    AVG(average_heartrate)                AS avg_hr
                FROM {schema}.activities
                WHERE LOWER(TRIM(sport_type)) = 'run'
                  AND start_date >= NOW() - INTERVAL '{int(weeks)} weeks'
                GROUP BY week
                ORDER BY week DESC
            )
            SELECT
                week,
                runs_that_week,
                ROUND(weekly_miles::numeric, 2)       AS weekly_miles,
                ROUND(longest_run_miles::numeric, 2)  AS longest_run_miles,
                ROUND(avg_hr::numeric, 1)             AS avg_hr
            FROM weekly
        """
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()

            if not rows:
                return "No running data found for this period."

            total_miles   = sum(r["weekly_miles"] for r in rows)
            avg_weekly    = total_miles / len(rows)
            avg_frequency = sum(r["runs_that_week"] for r in rows) / len(rows)
            longest       = max(r["longest_run_miles"] for r in rows)

            summary = {
                "weeks_analyzed":           len(rows),
                "avg_weekly_miles":         round(avg_weekly, 2),
                "avg_runs_per_week":        round(avg_frequency, 1),
                "longest_recent_run_miles": round(longest, 2),
                "total_miles_in_period":    round(total_miles, 2),
                "weekly_breakdown":         list(rows),
            }
            return json.dumps(summary, default=str)
        except Exception as e:
            logger.error(f"get_training_baseline failed: {e}")
            return f"Error fetching baseline: {e}"

    return [get_strava_stats, get_activity_vibes, get_training_baseline]


# ── Main RAG entry point ───────────────────────────────────────────────────────

def run_rag_agent(user_prompt: str, schema: str) -> str:
    """
    Run the LangGraph ReAct agent for a specific tenant.
    A new agent is created per call so each user's tools are schema-scoped.
    The LLM and tool definitions are lightweight — this is not a bottleneck.
    """
    tools  = create_tools_for_schema(schema)
    model  = ChatOpenAI(model=CHAT_MODEL, temperature=0.4)
    agent  = create_react_agent(model, tools)

    response = agent.invoke({
        "messages": [
            ("system", system_prompt),
            ("user",   user_prompt),
        ]
    })
    return response["messages"][-1].content


# ── Local dev helper ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Local ingestion: provision a test schema and load a JSON file.
    Usage: python -m src.rag_helper
    """
    TEST_ATHLETE_ID = os.getenv("TEST_ATHLETE_ID", "0")
    json_path       = "data/real_strava_data.json"

    schema = get_schema_name(TEST_ATHLETE_ID)
    provision_tenant_schema(TEST_ATHLETE_ID)

    try:
        with open(json_path) as f:
            activities = json.load(f)
        load_data_for_user(TEST_ATHLETE_ID, activities, schema)
    except FileNotFoundError:
        print(f"File not found: {json_path}")
    except json.JSONDecodeError:
        print(f"Invalid JSON: {json_path}")

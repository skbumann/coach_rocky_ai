-- Enable vector extension for semantic search w/ pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- MAIN ACTIVITIES TABLE
-- Structured fields as typed columns --> fast filtering
-- Unstructured text stored + embedded --> semantic search
-- Full JSON stored --> flexible access

CREATE TABLE activities (
    -- Primary identity
    id                    BIGINT PRIMARY KEY,
    upload_id             BIGINT,
    external_id           TEXT,

    -- Athlete reference
    athlete_id            BIGINT NOT NULL,

    -- Unstructured text 
    name                  TEXT,         
    description           TEXT,      

    -- Structured activity metrics 
    activity_type         TEXT,        
    sport_type            TEXT,
    distance_meters       NUMERIC,
    moving_time_seconds   INT,
    elapsed_time_seconds  INT,
    total_elevation_gain  NUMERIC,
    average_speed         NUMERIC,
    max_speed             NUMERIC,
    average_heartrate     NUMERIC,
    max_heartrate         NUMERIC,
    calories              NUMERIC,
    pr_count              INT,
    achievement_count     INT,
    kudos_count           INT,

    -- Timestamps
    start_date            TIMESTAMPTZ,
    start_date_local      TIMESTAMPTZ,
    timezone              TEXT,

    -- Gear & flags
    gear_id               TEXT,
    trainer               BOOLEAN DEFAULT FALSE,
    commute               BOOLEAN DEFAULT FALSE,
    private               BOOLEAN DEFAULT FALSE,

    -- Full raw document (for retrieval augmentation context)
    raw_json              JSONB NOT NULL,

    -- Timestamps for record management
    created_at            TIMESTAMPTZ DEFAULT NOW(),
    updated_at            TIMESTAMPTZ DEFAULT NOW()
);

-- EMBEDDINGS TABLE
-- Stores vector embeddings of the unstructured text

CREATE TABLE activity_embeddings (
    id              SERIAL PRIMARY KEY,
    activity_id     BIGINT NOT NULL REFERENCES activities(id) ON DELETE CASCADE,

    -- Which text was embedded (name, description, or combined)
    chunk_type      TEXT NOT NULL CHECK (chunk_type IN ('name', 'description', 'combined')),

    -- The actual text that was embedded (for inspection / debugging)
    chunk_text      TEXT NOT NULL,

    -- The embedding vector (1536 dims for OpenAI text-embedding-3-small)
    embedding       vector(1536),

    -- Model used to generate this embedding
    embedding_model TEXT NOT NULL DEFAULT 'text-embedding-3-small',

    created_at      TIMESTAMPTZ DEFAULT NOW(),

    -- Protects us later against adding duplicates
    UNIQUE (activity_id, chunk_text)
);

-- INDEXES

-- Vector similarity search index (IVFFlat for large datasets)
CREATE INDEX idx_embeddings_vector
    ON activity_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);  

-- Metadata filtering indexes (pre-filter before vector search)
CREATE INDEX idx_activities_athlete    ON activities(athlete_id);
CREATE INDEX idx_activities_type       ON activities(activity_type);
CREATE INDEX idx_activities_start_date ON activities(start_date);
CREATE INDEX idx_activities_distance   ON activities(distance_meters);
CREATE INDEX idx_activities_chunk_type ON activity_embeddings(chunk_type);

-- Full-text search index on name/description (alternative/complement to vector)
CREATE INDEX idx_activities_fts ON activities
    USING gin(to_tsvector('english', coalesce(name,'') || ' ' || coalesce(description,'')));

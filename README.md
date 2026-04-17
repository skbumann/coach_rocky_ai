# Coach Rocky (Under Construction)
Make your own personal coach RAG agent informed by your historical Strava data

# Tl;dr

## Why RAG?

Modern LLM tools don't have access to private or proprietary data. RAG is a way of working with this data without worrying about things like context window limits.

## Why this project?

I love Strava and I use it religiously. As such, my account has a very comprehensive picture of my athletic abilities, schedule, and even mood through unstructured data like captions and activity descriptions. With this tool, users will be able to retrieve their own Strava data through the API and ask an LLM general questions like training advice, summarizing specific stats, etc.

## Current Infrastructure

  -Data: Users are granted access to their Strava data (JSON) via the API, which is loaded into a PostgresSQL DB. Both the relational data and vector embeddings of the activity names are stored.

  -Tool-calling: A LangGraph ReAct agent answers natural language questions about the Strava-like training data by combining three tools:

    get_strava_stats — Runs SQL queries against activities database for structured metrics (distance totals, heart rate, date ranges, etc.)
    get_activity_vibes — Performs semantic/vector search to find activities by mood or feel (e.g. "when did I feel strong?", "sore legs")
    get_training_baseline — Computes weekly mileage, run frequency, and long run history to ground any training or safety advice in real context
    
  -FastAPI backend
  
  -Containerization via Docker for portability

## Future Work
  
  -Deploy on AWS EC2. This app is currently in the "Development" stage, which is limited to single users. 

  -Smarter prompting. It's pretty lean right now.
  
  -Adding memory to conversations
  
  -Anti-hallucination testing via RAGAS or DeepEval

  -UI/UX improvements

## Getting started (locally, due to Strava's API athlete limit)
1. Follow these instructions to make your own API application: https://developers.strava.com/docs/getting-started/. Set your website and "authorization callback domain" to "localhost."
2. Configure your .env file with your OPENAI_API_KEY, CLIENT_ID (from your Strava API application), and CLIENT_SECRET (from your Strava API application).
3. Follow the instructions in https_hack.txt. Strava requires a secure https url to redirect to after authorization so this provides a workaround when building locally.
4. Using docker, run:

```docker-compose up --build```

## Some (cherry-picked) examples 
<img width="1258" height="760" alt="strava_coach_agent" src="https://github.com/user-attachments/assets/7bc1d36b-d86c-4b23-aaf9-627e5ba3ac73" />

<img width="1264" height="765" alt="mood" src="https://github.com/user-attachments/assets/243dc609-f1c0-4c67-a209-7ae6bf997bd1" />


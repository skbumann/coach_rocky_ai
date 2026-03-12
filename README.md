# My Strava Coach (Under Construction!)
A Personal Coach RAG Agent Informed by Historical Exercise Data

# Tl;dr

## Why RAG?

Modern LLM tools don't have access to private or proprietary data. RAG is a way of working with this data without worrying about things like context window limits.

## Why this project?

I love Strava and I use it religiously. As such, my account has a very comprehensive picture of my athletic abilities, schedule, and even mood through unstructured data like captions and activity descriptions.
I am currently developing a tool that allows someone to sync their account (to-do), retrieve their data through the Strava API, and ask an LLM general questions like training advice, summarizing specific stats, etc.

## Current Infrastructure

  -Data: synthetic running, biking, and swimming activity data following the JSON format used by the Strava API (https://developers.strava.com/playground/), for a single athlete

  -Tool-calling: A LangGraph ReAct agent answers natural language questions about the Strava-like training data by combining three tools:

    get_strava_stats — Runs SQL queries against activities database for structured metrics (distance totals, heart rate, date ranges, etc.)
    get_activity_vibes — Performs semantic/vector search to find activities by mood or feel (e.g. "when did I feel strong?", "sore legs")
    get_training_baseline — Computes weekly mileage, run frequency, and long run history to ground any training or safety advice in real context
    
  -FastAPI backend
  
  -Containerization via Docker for portability

## Future Work

  -Enable Strava API access with OAuth 2.0
  
  -Deploy on AWS EC2 for cloud access
  
  -Adding memory to conversations
  
  -Anti-hallucination testing via RAGAS or DeepEval

  -UI/UX improvements

## Getting started
Just run:

```docker-compose up --build```

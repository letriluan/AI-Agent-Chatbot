# 🚚 Delivery Cadet Challenge – AI-Powered Data Exploration Agent

## Overview
This project implements an **AI-powered data exploration agent** capable of answering natural language questions over a structured relational database.

The agent dynamically:
- Interprets user questions
- Generates SQL queries
- Executes them safely against a local database
- Returns conversational answers
- Produces visualizations when requested

The system is designed to be **dataset-agnostic**, **safe**, and **easy to extend**.

---

## Key Capabilities
- Automatic database setup from CSV files  
- Primary key and foreign key inference  
- LangGraph-based agent orchestration  
- Safe, read-only SQL execution  
- Privacy guardrails for personal data  
- Optional chart generation (bar / line)  
- ChatGPT-style user interface  
- LangSmith execution tracing  

---

## Architecture Summary

### High-Level Flow
1. User asks a question via the chat UI  
2. LangGraph agent:
   - Inspects the database schema  
   - Generates SQL  
   - Executes SQL using a guarded tool  
   - Retries on SQL errors if needed  
   - Decides whether a visualization is required  
   - Produces a conversational answer  
3. The result is returned to the UI in real time  

### Technology Stack
- Python  
- SQLite  
- LangGraph  
- LangChain + Groq  
- FastAPI  
- Matplotlib  
- LangSmith  

---

## Repository Structure

```text
.
├── load_db.py                 # Database loader
├── data/                      # Provided CSV datasets
├── db/                        # SQLite database
├── src/
│   ├── agent/
│   │   ├── graph.py           # LangGraph agent definition
│   │   ├── schema.py          # Schema inspection utilities
│   │   ├── sql_tools.py       # Safe SQL execution tool
│   │   ├── privacy.py         # Privacy & PII guardrails
│   │   ├── plot_tools.py      # Visualization utilities
│   │   └── prompts.py         # Dataset-agnostic prompts
│   ├── server/
│   │   └── app.py             # FastAPI server + UI integration
│   └── ui/
│       └── index.html         # User interface
├── requirements.txt
└── README.md
```

---

## Set up and how to run

### Install Dependencies

```text
pip install -r requirements.txt
```

### Set up
Rebuild the database if you make changes to ```textload_db.py```:
```text
python load_db.py --data data --db db/app.sqlite
```

### LangSmith Tracing 
LangSmith tracing  is enabled via environment variables:
```text
GROQ_API_KEY=YOUR_KEY_HERE
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=YOUR_KEY_HERE
LANGCHAIN_PROJECT=<name_of_project>
GROQ_MODEL=<name_of_model> 
```
When the agent runs locally, all execution steps are automatically logged and visualized in LangSmith.

### Run the Application

Start the Server: 
```text
uvicorn src.server.app:app --port 8000
```

Open the Web Interface: 
```text
http://127.0.0.1:8000
```

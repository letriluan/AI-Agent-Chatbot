# 🤖 AI-Powered SQL Agent for E-commerce Data Analysis

## Overview
This project implements an **AI-powered SQL agent** that allows users to explore and analyze structured relational data using natural language.

The system automatically:
- Understands user question
- Inspects the database schema at runtime
- Generates safe, read-only SQL
- Executes queries securely
- Returns clear analytical answers
- Produces charts and visualizations when requested

The architecture is designed to be **dataset-agnostic**, **safe**, and **easy to extend**.

---


## Dataset Summary

This project uses the **[Olist Brazilian E-commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)** , a real-world dataset containing transactional data from a large online marketplace in Brazil.

The dataset represents the full lifecycle of e-commerce orders, from customer purchase to delivery and review, making it suitable for realistic analytical and business intelligence queries.

Dataset Contents:
- The dataset is provided as multiple relational tables, including:
- Customers – customer identifiers and location information (city, state)
- Orders – order status and timestamps (purchase, approval, delivery)
- Order Items – products purchased per order, prices, and freight costs
- Payments – payment methods, installment counts, and payment values
- Reviews – customer review scores and feedback
- Products – product metadata and category information
- Sellers – seller identifiers and location details
- Geolocation – geographic coordinates for Brazilian ZIP codes
- Category Translation – mapping between Portuguese and English product category names

These tables are connected through primary and foreign keys, forming a realistic relational schema for SQL-based data analysis.

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
├── load_db.py                 # CSV → SQLite database loader
├── data/                      # Dataset files (not committed)
├── db/                        # SQLite database & cache (not committed)
├── src/
│   ├── agent/
│   │   ├── graph.py           # Agent orchestration logic
│   │   ├── schema.py          # Schema inspection utilities
│   │   ├── sql_tools.py       # Safe SQL execution
│   │   ├── cache.py           # Multi-level caching
│   │   ├── privacy.py         # PII redaction
│   │   ├── plot_tools.py      # Visualization utilities
│   │   └── prompts.py         # Dataset-agnostic prompts
│   ├── server/
│   │   └── app.py             # FastAPI server
│   └── ui/
│       └── index.html         # Web UI
├── requirements.txt
├── .env.example
└── README.md
```
---

## Set up and how to run

### Install Dependencies

```text
pip install -r requirements.txt
```

### Build the Database
Rebuild the database if you make changes to ```textload_db.py```:
```text
python load_db.py --data data --db db/app.sqlite
```

### Environment Variables
```text
# LLM
GROQ_API_KEY=YOUR_API_KEY
GROQ_MODEL=llama-3.3-70b-versatile

# Database
DB_PATH=db/app.sqlite
CACHE_DB=db/cache.sqlite
CACHE_TTL_SECONDS=3600

# SQL safety
SQL_MAX_ROWS=50
SQL_TIMEOUT_SECONDS=2.0

# Windows plotting workaround
KMP_DUPLICATE_LIB_OK=TRUE
```


### Run the Application

Start the Server: 
```text
uvicorn src.server.app:app --port 8000
```

Open the Web Interface: 
```text
http://127.0.0.1:8000
```

# Simple NLP Answering System

A minimal API service that answers natural-language questions about member messages using retrieval-augmented generation (RAG).

## 🚀 Live Deployment

**URL:** `https://simple-nlp-answering-system.up.railway.app/`

**Quick Test:**
```bash
# Health check
curl https://simple-nlp-answering-system.up.railway.app/

# Ask a question
curl -X POST https://simple-nlp-answering-system.up.railway.app/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many cars does Vikram Desai have?"}'
```

**Example Questions:**
- "When is Layla planning her trip to London?"
- "How many cars does Vikram Desai have?"
- "What are Amina's favorite restaurants?"

## Features

- **🔍 Semantic Search**: Uses FastEmbed (BAAI/bge-small-en-v1.5) to embed queries and messages for accurate similarity-based retrieval
- **👤 Smart Name Recognition**: spaCy NER (en_core_web_md) extracts person names from queries, with fuzzy matching and first-name disambiguation
- **📊 Contextual Filtering**: Retrieves top-K most relevant messages filtered by detected member names for focused results
- **🧠 LLM Generation**: Groq API (llama-3.3-70b-versatile) synthesizes natural language answers from retrieved context
- **⚡ Fast Cold Starts**: Models pre-initialized at startup (~20s) for sub-3s response times on first request
- **🔒 Rate Limiting**: Built-in request throttling (5 req/min per IP) to prevent abuse
- **📦 Production-Ready**: Docker containerized with persistent model caching and health checks
- **🌐 CORS Enabled**: Cross-origin requests supported for frontend integration
- **📝 Structured Prompts**: Custom system/user prompts loaded from files for easy tuning


## Architecture

- **Data Ingestion**: Fetches messages from the November7 API (`GET /messages`) and stores locally in JSON format with embeddings
- **Vector Store**: Messages are embedded using FastEmbed (`BAAI/bge-small-en-v1.5`, 384 dims) and indexed in Pinecone with cosine similarity
- **Retrieval**: Two-stage process: (1) spaCy NER extracts person entities from query, (2) semantic search in Pinecone filtered by member name
- **Name Resolution**: Cached `known_names.json` provides fuzzy matching, first-name disambiguation, and "Did you mean?" suggestions
- **Generation**: Groq API (llama-3.3-70b-versatile) receives chronologically sorted context with metadata (member name, latest activity, snippet count)
- **API**: FastAPI service with async lifespan for model pre-initialization, structured logging, and comprehensive error handling

## Design Notes
### Alternative Approaches
1. **Creating User Profiles**:
   - Ideally, the question answers could be improved by maintaining a user profile database with nicknames and list of facts (e.g., favorite foods, hobbies). However, this would require much more setup (e.g., defining a schema for the profile database or defining an llm prompt to summarize the messages), so I opted for an approach using NER and RAG based message retrieval.
   - If I had created a user profile database, I'd expose it via an MCP tool (client-initiated call into the profile service) or pipe the question through text-to-SQL so the agent can hit the structured database directly.
2. **Improving Retrieval**:
   - I considered building an agentic system where the first step would be to generate multiple alternative questions to improve lookup accuracy in Pinecone. However, for such a small dataset (and such short messages), this would likely add unnecessary complexity and latency without significant benefit.
3. **Summary timeline database of user trips**:
   - I thought about creating a structured timeline database summarizing user trips extracted from messages (creating a timeline/history database) to facilitate more accurate date-related queries. However, this would require significant upfront processing and might not cover all edge cases, so I decided to rely on the existing message data with enhanced prompt engineering instead.
4. **Categorizations of messages and query**:
   - I considered categorizing messages (e.g., travel plans, purchases, dining) and classifying user queries to route them to specialized retrieval/generation pipelines. However, this would add complexity and quite a bit of upfront processing, so I opted for a single unified approach with improved prompt instructions.
5.  **Keyword-Based Name Matching**:
    - I considered creating a simple `Set` of all unique member names (e.g., "Vikram Desai") on startup. When a query came in, the service would iterate through this set and check if any name was a substring of the question.
    - **Reason for Choosing NER Instead:** This simple keyword-matching approach is fast but very brittle. It would fail on common and crucial variations, particularly possessives (e.g., it wouldn't match "Layla" in "Layla's trip"). A statistical NER model, while not perfect, is trained to be context-aware and can correctly identify "Layla" as a `PERSON` entity from "Layla's," making it a more robust and flexible solution out-of-the-box.


### Data Insights
3,349 messages from 10 members over 364 days (Nov 2024 - Nov 2025)

**Data Quality:**
- No missing timestamps, duplicate IDs, or malformed data
- All timestamps are UTC-aware and within valid range
- Zero duplicate messages (100% unique content)
- 2 extremely short messages (<10 chars) - appear to be incomplete sentences
   - "I want to" and "I finally"
- 48 long messages (>88 chars) - mostly from Amina Van Den Berg requesting travel/concierge services

**Message Patterns:**
- Average message length: 68 characters (median: 68, std: 8.7)
- Message frequency: ~26 hours between messages (median: 18 hours)
- Fastest burst: 1.13 minutes between consecutive messages
- Longest gap: ~7 days between messages
- Uneven distribution: 288-365 messages per user (Lorenzo Cavalli has fewest, Lily O'Sullivan has most)

**Notable Observations:**
- All users have nearly identical message counts (~334 avg)
- No duplicate message content despite high volume, indicating carefully curated dataset
- Message timing shows realistic human patterns with variable gaps
- From visual inspection, messages generally appear to be about scheduling restaurant visits (Michelin star), travel plans, and car service requests, updating profile information / preferences (e.g., favorite cuisines, phone numbers, insurance info, etc.), all of which generally align with what a concierge service might build.


## Local Development

### Prerequisites
- Python 3.10-3.12
- Pinecone account with an index created (`simple-nlp`, cosine similarity, 384 dimensions)
- Groq account with an API key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/simple_nlp_answering_system.git
   cd simple_nlp_answering_system
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   ```bash
   export PINECONE_API_KEY="your-pinecone-key"
   export GROQ_API_KEY="your-groq-key"
   export GROQ_MODEL="llama-3.3-70b-versatile"
   ```

5. **Prepare data** (one-time setup)
   ```bash
   # Fetch messages from the API and store in data/all_messages.json
   python run_one_time/get_messages.py
   
   # Generate embeddings and upload to Pinecone
   python run_one_time/pinecone_upload.py

   # Cache normalized member names for validation in config/known_names.json
   python run_one_time/get_known_names.py
   ```

6. **Start the API server**
   ```bash
   uvicorn main:app --reload
   ```
   The server will start on `http://localhost:8000`. First startup takes ~20s to download and cache the spaCy model.

7. **Test the service**
   ```bash
   # Health check
   curl -s http://localhost:8000/
   
   # Ask a question
   curl -X POST http://localhost:8000/ask \
      -H "Content-Type: application/json" \
      -d '{"question": "When is Layla planning her trip to London?"}'
   ```

## Deployment

### Option 1: Railway (Recommended)

1. Push your code to GitHub
2. Visit [railway.app](https://railway.app) and create a new project
3. Connect your GitHub repository
4. Add environment variables in the Railway dashboard:
   - `PINECONE_API_KEY`
   - `GROQ_API_KEY`
   - `GROQ_MODEL` (e.g., `llama-3.3-70b-versatile`)
5. Railway will automatically detect the Dockerfile and deploy. The spaCy model is downloaded during Docker build.
6. **Verify the live service**
    ```bash
    # Health check (replace with your Railway domain)
    curl https://your-app.up.railway.app/

    # Ask a question
    curl -X POST https://your-app.up.railway.app/ask \
       -H "Content-Type: application/json" \
       -d '{"question": "When is Layla planning her trip to London?"}'
    ```
    Find your domain under Railway Settings → Domains.

### Option 2: Render

1. Push your code to GitHub
2. Visit [render.com](https://render.com) and create a new Web Service
3. Connect your repository
4. Render will use `render.yaml` for configuration
5. Add environment variables in the Render dashboard
   - `PINECONE_API_KEY`
   - `GROQ_API_KEY`
   - `GROQ_MODEL`
6. Deploy

### Option 3: Docker (Manual)

```bash
# Build the image
docker build -t nlp-qa-service .

# Run locally
docker run -p 8000:8000 \
   -e PINECONE_API_KEY="your-key" \
   -e GROQ_API_KEY="your-groq-key" \
   -e GROQ_MODEL="llama-3.3-70b-versatile" \
  nlp-qa-service
```

## API Documentation

Once deployed, visit `https://simple-nlp-answering-system.up.railway.app/docs` for interactive Swagger documentation.

### Endpoints

**POST /ask**
- Rate limited to 5 requests/minute per IP
- Request body: `{"question": "your question here"}`
- Response: `{"answer": "the generated answer"}`

**GET /**
- Health check endpoint
- Returns: `{"status": "ok", "message": "Q&A Service is running."}`

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `PINECONE_API_KEY` | Yes | Your Pinecone API key |
| `GROQ_API_KEY` | Yes | Groq API key for generation |
| `GROQ_MODEL` | Yes | Groq model name (default `llama-3.3-70b-versatile`) |
| `QA_CONFIG_PATH` | No | Path to config file (default: `config/config.yaml`) |
| `SPACY_MODEL_DIR` | No | Override for spaCy model storage (default: `./runtime_models/spacy`) |

## spaCy Model Handling

The service uses `en_core_web_md` (~90MB in memory) for name extraction, providing good accuracy while fitting Railway's free tier memory limits.

**How it works:**
- **Docker build**: Model is downloaded during `docker build` and baked into the image (no runtime download)
- **Local development**: First startup downloads model to `./runtime_models/spacy` (~20s), then caches it for instant subsequent runs
- **Production**: Model is pre-loaded during container startup (FastAPI lifespan) for fast first-request response (~3s)

## Project Structure

```
├── main.py                 # FastAPI application
├── config/
│   └── config.yaml        # Configuration (Pinecone index, embedder, etc.)
├── src/
│   ├── rag/
│   │   ├── retriever.py   # Semantic search and NER filtering
│   │   └── service.py     # QA orchestration
│   └── utils.py           # Logging and config utilities
├── run_one_time/
│   ├── get_messages.py    # Fetch data from API
│   ├── pinecone_upload.py # Index messages in Pinecone
│   └── get_known_names.py # Cache normalized member names
├── Dockerfile             # Container definition
└── requirements.txt       # Python dependencies
```



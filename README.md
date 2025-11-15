# Simple NLP Answering System

A minimal API service that answers natural-language questions about member messages using retrieval-augmented generation (RAG).

**Live Deployment:** [Add your URL here after deployment]

## Architecture

- **Data Ingestion**: Fetches messages from the November7 API (`GET /messages`) and stores locally
- **Vector Store**: Messages are embedded using FastEmbed (`BAAI/bge-small-en-v1.5`) and indexed in Pinecone
- **Retrieval**: Uses semantic search with Name Entity Based (NER) based filtering plus a cached `known_names.json` lookup to validate/auto-complete member names before querying Pinecone vector database
- **Generation**: Groq API generates natural language answers from context
- **API**: FastAPI service with rate limiting

## Local Development

### Prerequisites
- Python 3.11+
- Pinecone account with an index created
- Groq account with an API key

### Setup

1. **Install dependencies**
   ```bash
   pip install -e .
   ```

2. **Set environment variables**
   ```bash
   export PINECONE_API_KEY="your-pinecone-key"
   export GROQ_API_KEY="your-groq-key"
   export GROQ_MODEL="llama-3.3-70b-versatile"
   ```

3. **Prepare data** (one-time setup)
   ```bash
   # Fetch messages from the API
   python run_one_time/get_messages.py
   
   # Upload to Pinecone
   python run_one_time/pinecone_upload.py

   # Cache normalized member names for validation
   python run_one_time/get_known_names.py
   ```

4. **Start the API server**
   ```bash
   uvicorn main:app --reload
   ```

5. **Test locally**
   ```bash
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
   - `GROQ_MODEL`
5. Create a volume in Railway and mount it (for example to `/data`). Set `SPACY_MODEL_DIR=/data/spacy` so the large `en_core_web_lg` model is cached between deployments.
6. Railway will automatically detect the Dockerfile and deploy

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

Once deployed, visit `https://your-domain.com/docs` for interactive Swagger documentation.

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
| `SPACY_MODEL_DIR` | No | Directory (preferably on a persistent volume) where `en_core_web_lg` will be cached |

## spaCy model caching

The service requires `en_core_web_lg` for high-quality name extraction. Instead of baking the 600 MB wheel into the Docker image, the app now downloads the model at runtime into `runtime_models/spacy` (or the directory specified via `SPACY_MODEL_DIR`).

- **Local development**: the first run will download the model into `./runtime_models/spacy`. Delete that folder to force a refresh.
- **Railway / containers**: mount a persistent volume and point `SPACY_MODEL_DIR` to that path so the model is downloaded once and reused across deployments. Without a volume, the model will be fetched on each cold start, adding ~1 minute to boot time.

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


## Design Notes
### Alternative Approaches Considered
1. **Creating User Profiles**:
   - Ideally, the simple name handling could be improved by maintaining a user profile database with nicknames and list of facts (e.g., favorite foods, hobbies). However, this would require additional data that likely wouldn't be able to be extracted from user messages along, so I opted for an approach using NER and cached names.
   - To query the user profile database, I'd expose it via an MCP tool (client-initiated call into the profile service) or pipe the question through text-to-SQL so the agent can hit the structured database directly with low latency.
2. **Improving Retrieval**:
   - I considered building an agentic system where the first step would be to generate multiple alternative questions to improve lookup accuracy in Pinecone. However, for such a small dataset (and such short messages), this would likely add unnecessary complexity and latency without significant benefit.
3. **Summary timeline database of user trips**:
   - I thought about creating a structured timeline database summarizing user trips extracted from messages to facilitate more accurate date-related queries. However, this would require significant upfront processing and might not cover all edge cases, so I decided to rely on the existing message data with enhanced prompt engineering instead.
4. **Categorizations of messages and query**:
   - I considered categorizing messages (e.g., travel plans, purchases, dining) and classifying user queries to route them to specialized retrieval/generation pipelines. However, this would add complexity and quite a bit of upfront processing, so I opted for a single unified approach with improved prompt instructions.


### Data Insights


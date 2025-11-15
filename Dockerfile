# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for spaCy and sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Configure spaCy cache directory.
# The app downloads en_core_web_lg at runtime into this path if it is missing.
# In production, override SPACY_MODEL_DIR to a mounted persistent volume so the
# 600MB model is fetched only once (Railway: mount /data and set SPACY_MODEL_DIR=/data/spacy).
ENV SPACY_MODEL_DIR=/app/runtime_models/spacy

# Copy application code
COPY . .

# Expose port (Railway/Render will use $PORT env var)
EXPOSE 8000

# Run the FastAPI app with uvicorn
# Wrap in sh -c so ${PORT:-8000} is interpreted at runtime.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

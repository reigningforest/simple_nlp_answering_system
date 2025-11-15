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

# Copy application code
COPY . .

# Expose port (Railway/Render will use $PORT env var)
EXPOSE 8000

# Run the FastAPI app with uvicorn
# Use $PORT environment variable with fallback to 8000
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}

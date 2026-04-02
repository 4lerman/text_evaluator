FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV HF_HOME /models/huggingface
ENV SPACY_DATA /models/spacy

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models to bake into the image
# This ensures fast startup and local availability
RUN python -m spacy download xx_sent_ud_sm && \
    python -m spacy download en_core_web_sm && \
    python -m spacy download ru_core_news_sm

# Pre-download SentenceTransformer (intfloat/multilingual-e5-large-instruct)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large-instruct')"

# Copy project files
COPY app/ ./app/
COPY frontend/ ./frontend/

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

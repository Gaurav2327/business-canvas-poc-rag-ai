# Multi-stage Dockerfile for AWS ECS with Ollama + Python Backend
# This creates a complete container with both Ollama and the RAG backend

FROM python:3.11-slim as backend-base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model to speed up startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
COPY backend/server.py .

# ==========================================
# Final stage with Ollama + Backend
# ==========================================
FROM ubuntu:22.04

WORKDIR /app

# Install dependencies for both Python and Ollama
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy Python environment from backend-base
COPY --from=backend-base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend-base /usr/local/bin /usr/local/bin
COPY --from=backend-base /root/.cache /root/.cache

# Copy application
COPY --from=backend-base /app/server.py .

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting Ollama service..."\n\
ollama serve &\n\
OLLAMA_PID=$!\n\
\n\
echo "Waiting for Ollama to be ready..."\n\
for i in {1..30}; do\n\
  if curl -s http://localhost:11434/api/tags > /dev/null; then\n\
    echo "Ollama is ready!"\n\
    break\n\
  fi\n\
  echo "Waiting for Ollama... ($i/30)"\n\
  sleep 2\n\
done\n\
\n\
echo "Pulling Llama3 model..."\n\
ollama pull ${OLLAMA_MODEL:-llama3} || echo "Model pull failed, will retry on demand"\n\
\n\
echo "Starting Python backend..."\n\
exec python server.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose ports
EXPOSE 3000 11434

# Environment variables
ENV PORT=3000
ENV OLLAMA_HOST=http://localhost:11434
ENV OLLAMA_MODEL=llama3
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Start both services
CMD ["/app/start.sh"]


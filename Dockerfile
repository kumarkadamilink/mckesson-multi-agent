# ──────────────────────────────────────────────
# McKesson Multi-Agent Procurement Workflow
# Hosted Agent container for Azure AI Foundry
# ──────────────────────────────────────────────
FROM python:3.11-slim

# System deps needed by onnxruntime and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent source files
COPY AgentsWithRouterSynthesizer_Procurement.py .
COPY local_search_agent.py .
COPY index_documents.py .
COPY server.py .

# Copy the pre-built ChromaDB index and contract documents
COPY data/ ./data/

# Expose port expected by Foundry hosted agent runtime
EXPOSE 8088

# Start the FastAPI server
CMD ["python", "server.py"]

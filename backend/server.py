#!/usr/bin/env python3
"""
RAG Backend Server - Python Version
FREE version using Ollama (local LLM) + Local Embeddings
100% FREE - No API costs!
"""

import os
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAG Backend - Python", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PORT = int(os.getenv("PORT", "3000"))
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

print("🔍 Using FREE providers:")
print("  Embedding: Local (sentence-transformers/all-MiniLM-L6-v2)")
print(f"  Generation: Ollama ({OLLAMA_MODEL})")
print(f"  Ollama Host: {OLLAMA_HOST}")

if not PINECONE_API_KEY:
    print("❌ Missing PINECONE_API_KEY. Check .env")
    exit(1)

# Initialize clients
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Initialize local embedding model (lazy loading)
embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    """Get or initialize the embedding model (lazy loading)."""
    global embedding_model
    if embedding_model is None:
        print("📥 Loading local embedding model (first time may take 1-2 minutes)...")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Embedding model loaded and cached!")
    return embedding_model


# ---------------------- Request/Response Models ----------------------

class IndexRequest(BaseModel):
    text: str
    source: Optional[str] = "manual"
    clearPrevious: Optional[bool] = False


class QueryRequest(BaseModel):
    query: str
    filterBySource: Optional[str] = None


class IndexResponse(BaseModel):
    ok: bool
    indexedChunks: int
    indexName: str
    clearedPrevious: bool


class RetrievedDoc(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    retrieved: List[RetrievedDoc]


# ---------------------- Helper Functions ----------------------

def chunk_text(text: str, max_len: int = 2000) -> List[str]:
    """Split large text into chunks with smart splitting."""
    chunks = []
    
    if len(text) <= max_len:
        return [text]
    
    # Split by paragraphs
    paragraphs = [p for p in text.replace('\r\n', '\n').split('\n') if p.strip()]
    current_chunk = ''
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 > max_len:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # If single paragraph is too long, split it further
            if len(para) > max_len:
                start = 0
                while start < len(para):
                    piece = para[start:start + max_len]
                    if piece.strip():
                        chunks.append(piece.strip())
                    start += max_len
                current_chunk = ''
            else:
                current_chunk = para
        else:
            current_chunk += ('\n' if current_chunk else '') + para
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out very small chunks (unless it's the only chunk)
    filtered_chunks = [c for c in chunks if len(c) >= 50] if len(chunks) > 1 else chunks
    
    print(f"📦 Split text ({len(text)} chars) into {len(filtered_chunks)} chunks")
    return filtered_chunks if filtered_chunks else [text]


def embed_text_local(text: str) -> List[float]:
    """Generate embeddings locally using sentence-transformers (FREE)."""
    try:
        model = get_embedding_model()
        
        # Truncate if too long (model handles this internally, but we do it explicitly)
        truncated = text[:2000]
        
        # Generate embeddings
        embedding = model.encode(truncated, normalize_embeddings=True)
        
        # Convert to list
        return embedding.tolist()
    except Exception as e:
        print(f"❌ Local embedding error: {e}")
        raise


async def generate_ollama(prompt: str, max_tokens: int = 512) -> str:
    """Generate RAG answer with Ollama (FREE)."""
    print(f"🤖 Calling Ollama: {OLLAMA_MODEL}")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": max_tokens,
                    }
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Ollama returned {response.status_code}: {response.text}"
                )
            
            data = response.json()
            return data.get("response", "")
    except httpx.ConnectError:
        raise HTTPException(
            status_code=500,
            detail="Ollama is not running. Start it with: ollama serve"
        )
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------- API Endpoints ----------------------

@app.post("/index", response_model=IndexResponse)
async def index_text(request: IndexRequest):
    """Add (index) new text into Pinecone."""
    try:
        text = request.text
        source = request.source or "manual"
        clear_previous = request.clearPrevious or False
        
        if not text or len(text) < 10:
            raise HTTPException(status_code=400, detail="Text too short")
        
        print(f"📥 Indexing request: {len(text)} characters from {source}")
        
        # Get index
        index = pinecone_client.Index(PINECONE_INDEX)
        
        # 1️⃣ Clear previous content if requested
        if clear_previous:
            print("🗑️  Clearing previous content from index...")
            try:
                index.delete(delete_all=True)
                print("✅ Previous content cleared")
            except Exception as clear_err:
                print(f"Warning: Could not clear index: {clear_err}")
        
        # 2️⃣ Chunk text
        chunks = chunk_text(text, 2000)
        
        # 3️⃣ Embed and build vectors
        print("🔢 Generating embeddings locally...")
        vectors = []
        timestamp = int(time.time() * 1000)
        
        for i, chunk in enumerate(chunks):
            embedding = embed_text_local(chunk)
            vectors.append({
                "id": f"{timestamp}-{i}",
                "values": embedding,
                "metadata": {
                    "source": source,
                    "text": chunk
                }
            })
            
            if (i + 1) % 5 == 0:
                print(f"  Embedded {i + 1}/{len(chunks)} chunks...")
        
        print(f"✅ All {len(vectors)} chunks embedded!")
        
        # 4️⃣ Upsert into Pinecone
        index.upsert(vectors=vectors)
        
        return IndexResponse(
            ok=True,
            indexedChunks=len(vectors),
            indexName=PINECONE_INDEX,
            clearedPrevious=clear_previous
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Index error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query RAG pipeline."""
    try:
        query = request.query
        filter_by_source = request.filterBySource
        
        if not query:
            raise HTTPException(status_code=400, detail="No query provided")
        
        # 1️⃣ Embed query
        print("🔍 Embedding query locally...")
        q_vec = embed_text_local(query)
        
        # 2️⃣ Search in Pinecone
        index = pinecone_client.Index(PINECONE_INDEX)
        query_options = {
            "vector": q_vec,
            "top_k": 5,
            "include_metadata": True,
        }
        
        if filter_by_source:
            query_options["filter"] = {"source": {"$eq": filter_by_source}}
            print(f"🔍 Filtering results by source: {filter_by_source}")
        
        query_resp = index.query(**query_options)
        
        # Extract hits
        hits = [
            RetrievedDoc(
                id=match.get("id", ""),
                score=match.get("score", 0.0),
                metadata=match.get("metadata", {})
            )
            for match in query_resp.get("matches", [])
        ]
        
        if not hits:
            no_content_msg = (
                "I don't have any indexed content"
                + (" from this page" if filter_by_source else "")
                + " to answer your question. Please index some content first."
            )
            return QueryResponse(answer=no_content_msg, retrieved=[])
        
        # 3️⃣ Build context prompt
        context = "\n\n".join([
            f"Context {i + 1}:\n{hit.metadata.get('text', '')}"
            for i, hit in enumerate(hits)
        ])
        
        prompt = (
            f"You are a helpful assistant. Use ONLY the following context to answer the question. "
            f"If the answer is not in the context, say \"I don't know\".\n\n"
            f"{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        
        # 4️⃣ Generate answer with Ollama
        answer = await generate_ollama(prompt)
        
        return QueryResponse(answer=answer, retrieved=hits)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Ollama
        ollama_ok = False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{OLLAMA_HOST}/api/tags")
                ollama_ok = response.status_code == 200
        except:
            ollama_ok = False
        
        # Check embeddings
        embedding_ok = False
        try:
            embed_text_local("test")
            embedding_ok = True
        except:
            embedding_ok = False
        
        return {
            "status": "ok",
            "providers": {
                "ollama": "connected" if ollama_ok else "disconnected",
                "embeddings": "ready" if embedding_ok else "loading",
                "pinecone": "connected"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "🚀 FREE RAG Server (Python)",
        "version": "1.0.0",
        "providers": {
            "embedding": "Local (sentence-transformers)",
            "generation": f"Ollama ({OLLAMA_MODEL})",
            "vector_db": "Pinecone"
        }
    }


if __name__ == "__main__":
    import uvicorn
    print(f"🚀 FREE Server starting on http://localhost:{PORT}")
    print("💡 Using 100% free providers (Ollama + Local Embeddings)")
    print("📝 Make sure Ollama is running: ollama serve")
    uvicorn.run(app, host="0.0.0.0", port=PORT)


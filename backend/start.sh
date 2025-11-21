#!/bin/bash

# Start script for Python backend
# Usage: ./start.sh

echo "🚀 Starting Python RAG Backend..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created!"
    echo ""
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed!"
    echo ""
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "📝 Creating .env template..."
    cat > .env << EOF
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_INDEX=rag
PORT=3000
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
EOF
    echo "✅ .env template created!"
    echo "⚠️  Please update .env with your Pinecone API key"
    echo ""
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Warning: Ollama is not running!"
    echo "💡 Start Ollama in another terminal: ollama serve"
    echo ""
fi

echo "🎯 Starting server on port 3000..."
echo "📚 API docs available at: http://localhost:3000/docs"
echo "🏥 Health check: http://localhost:3000/health"
echo ""

# Start the server
python server.py


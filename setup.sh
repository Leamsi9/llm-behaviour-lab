#!/bin/bash
# Setup script for LLM Behaviour Lab

echo "=========================================="
echo "LLM Behaviour Lab - Setup"
echo "=========================================="
echo ""
echo "Prerequisites:"
echo "1. Install Ollama from https://ollama.ai/download"
echo "2. Pull models: ollama pull llama3:text, ollama pull llama3:instruct"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "Your current .env configuration:"
if [ -f ".env" ]; then
    cat .env
else
    echo "⚠️  No .env file found. Creating default..."
    cat > .env << 'EOF'
MAX_INPUT_LENGTH=12000
MAX_CONTEXT_TOKENS=8192
MAX_OUTPUT_TOKENS=4096
REQUEST_TIMEOUT=180.0
EOF
    echo "✓ Created .env with defaults for 32GB+ RAM system"
    cat .env
fi

echo ""
echo "=========================================="
echo "Next steps:"
echo "1. Start Ollama: ollama serve"
echo "2. Run the app:"
echo "   source venv/bin/activate"
echo "   uvicorn app_ollama:app --host 0.0.0.0 --port 8000 --reload"
echo "3. Open: http://localhost:8000/static/ui_multi.html"
echo "=========================================="

#!/bin/bash
# Download Qwen 2.5 14B models (Q4_K_M quantization)
# These are official GGUF files from Qwen team

set -e

echo "=========================================="
echo "Downloading Qwen 2.5 14B Models"
echo "=========================================="
echo ""
echo "This will download approximately 18GB of models:"
echo "  - Qwen 2.5 14B Base (~9GB)"
echo "  - Qwen 2.5 14B Instruct (~9GB)"
echo ""

cd models

# Download Base Model
echo "📥 Downloading Qwen 2.5 14B Base model..."
if [ -f "Qwen2.5-14B-Q4_K_M.gguf" ]; then
    echo "✓ Base model already exists, skipping..."
else
    wget -c https://huggingface.co/Qwen/Qwen2.5-14B-GGUF/resolve/main/qwen2.5-14b-q4_k_m.gguf \
         -O Qwen2.5-14B-Q4_K_M.gguf
    echo "✓ Base model downloaded successfully!"
fi

echo ""

# Download Instruct Model
echo "📥 Downloading Qwen 2.5 14B Instruct model..."
if [ -f "Qwen2.5-14B-Instruct-Q4_K_M.gguf" ]; then
    echo "✓ Instruct model already exists, skipping..."
else
    wget -c https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf \
         -O Qwen2.5-14B-Instruct-Q4_K_M.gguf
    echo "✓ Instruct model downloaded successfully!"
fi

echo ""
echo "=========================================="
echo "✓ Download Complete!"
echo "=========================================="
echo ""
echo "Models saved in: $(pwd)"
echo ""
echo "File sizes:"
ls -lh Qwen2.5-14B*.gguf 2>/dev/null || echo "No Qwen models found"
echo ""
echo "You can now restart your app to load these models."

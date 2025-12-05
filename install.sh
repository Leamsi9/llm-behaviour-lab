#!/bin/bash
# Setup script for LLM Behaviour Lab

echo "=========================================="
echo "LLM Behaviour Lab - Setup"
echo "=========================================="
echo ""
echo "Prerequisites:"
echo "1. Ollama installed locally (this script can help install it on Linux/macOS)."
echo "2. At least one model pulled (this script will offer to pull qwen3:0.6b by default)."
echo ""

OS_TYPE="$(uname -s 2>/dev/null || echo 'UNKNOWN')"

check_ollama() {
    if command -v ollama >/dev/null 2>&1; then
        echo "✓ Ollama detected: $(command -v ollama)"
        return 0
    fi

    echo "Ollama does not appear to be installed on this system."
    case "$OS_TYPE" in
        Darwin)
            echo "Detected macOS."
            read -r -p "Install Ollama via Homebrew now? [y/N] " REPLY
            if [[ "$REPLY" =~ ^[Yy]$ ]]; then
                brew install ollama || echo "⚠️ Homebrew install of Ollama failed. Please install it manually from https://ollama.ai/download."
            fi
            ;;
        Linux)
            echo "Detected Linux."
            read -r -p "Install Ollama via the official install script now? [y/N] " REPLY
            if [[ "$REPLY" =~ ^[Yy]$ ]]; then
                curl -fsSL https://ollama.ai/install.sh | sh || echo "⚠️ Ollama install script failed. Please install it manually from https://ollama.ai/."
            fi
            ;;
        *)
            echo "OS '$OS_TYPE' is not recognised for automatic Ollama installation."
            echo "Please install Ollama manually from https://ollama.ai/."
            ;;
    esac

    if command -v ollama >/dev/null 2>&1; then
        echo "✓ Ollama is now available."
        return 0
    else
        echo "⚠️ Ollama is still not available on PATH. You will need to install and run it separately."
        return 1
    fi
}

check_scaphandre() {
    if command -v scaphandre >/dev/null 2>&1; then
        echo "✓ Scaphandre detected: $(command -v scaphandre)"
        return 0
    fi

    echo "Scaphandre (optional, for per-process live power) does not appear to be installed."

    case "$OS_TYPE" in
        Linux)
            echo "Detected Linux."
            echo "This helper can download a Scaphandre Linux x86_64 binary from GitHub releases."
            read -r -p "Download and install a local Scaphandre binary now? [y/N] " REPLY
            if [[ "$REPLY" =~ ^[Yy]$ ]]; then
                INSTALL_DIR="$HOME/.local/bin"
                if [ ! -d "$INSTALL_DIR" ]; then
                    INSTALL_DIR="$(pwd)/.local-bin"
                    mkdir -p "$INSTALL_DIR"
                    echo "Creating local bin directory at $INSTALL_DIR."
                fi

                SCAPHANDRE_URL="https://github.com/hubblo-org/scaphandre/releases/latest/download/scaphandre-Linux-x86_64"
                TARGET="$INSTALL_DIR/scaphandre"

                echo "Downloading Scaphandre to $TARGET ..."
                if curl -fsSL "$SCAPHANDRE_URL" -o "$TARGET"; then
                    chmod +x "$TARGET"
                    echo "✓ Scaphandre installed at $TARGET"
                    case ":$PATH:" in
                        *":$INSTALL_DIR:"*)
                            echo "PATH already includes $INSTALL_DIR."
                            ;;
                        *)
                            echo "⚠️ Your PATH does not appear to include $INSTALL_DIR."
                            echo "   Add this to your shell rc (e.g. ~/.bashrc or ~/.zshrc):"
                            echo "   export PATH=\"$INSTALL_DIR:\$PATH\""
                            ;;
                    esac
                else
                    echo "⚠️ Failed to download Scaphandre from $SCAPHANDRE_URL."
                    echo "   You can also install it manually: https://github.com/hubblo-org/scaphandre/releases"
                fi
            fi
            ;;
        *)
            echo "Automatic Scaphandre install is only supported on Linux via this script."
            echo "Please see https://github.com/hubblo-org/scaphandre for install instructions."
            ;;
    esac

    if command -v scaphandre >/dev/null 2>&1; then
        echo "✓ Scaphandre is now available."
        return 0
    else
        echo "Scaphandre is still not available on PATH. Live per-process metrics will fall back to plain RAPL."
        return 1
    fi
}

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt

echo ""
echo "Python environment setup complete."

echo ""
echo "Checking for Ollama and a default test model (qwen3:0.6b)..."
if check_ollama; then
    if ollama list 2>/dev/null | grep -q 'qwen3:0.6b'; then
        echo "✓ Model 'qwen3:0.6b' already present."
    else
        echo "The default test model 'qwen3:0.6b' is not present."
        read -r -p "Pull qwen3:0.6b now? [Y/n] " REPLY
        if [[ -z "$REPLY" || "$REPLY" =~ ^[Yy]$ ]]; then
            ollama pull qwen3:0.6b || echo "⚠️ Failed to pull qwen3:0.6b. You can pull it manually later with 'ollama pull qwen3:0.6b'."
        else
            echo "Skipping model pull. You can run 'ollama pull qwen3:0.6b' later."
        fi
    fi
else
    echo "Skipping default model pull because Ollama is not currently available."
fi

echo ""
echo "Checking for Scaphandre (optional, for per-process live power)..."
check_scaphandre || true

echo ""
echo "✓ Setup complete!"

echo ""
echo "=========================================="
echo "Next steps:"
echo "1. Start Ollama: ollama serve"
echo "2. Run the integrated lab:"
echo "   source .venv/bin/activate"
echo "   uvicorn app_llm_behaviour_lab:app --host 0.0.0.0 --port 8001 --reload"
echo "3. Open: http://localhost:8001/energy"
echo "=========================================="

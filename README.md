# LLM Harness Lab

A FastAPI-based web application for comparing pre-RLHF and post-RLHF language models. This tool allows you to run and compare the outputs of different language models side by side, with a focus on analyzing the differences between base and instruction-tuned models.

## Features

- Side-by-side comparison of base and instruction-tuned models
- Real-time token streaming
- Adjustable generation parameters (temperature, max tokens, etc.)
- Token counting and performance metrics
- Web-based interface with responsive design

## Prerequisites

- Python 3.9+
- 16GB+ RAM (for 8B parameter models)
- 10GB+ free disk space for models
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-profiling.git
   cd ai-profiling
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install fastapi uvicorn python-multipart jinja2 llama-cpp-python
   ```

## Downloading Models

1. Create a `models` directory:
   ```bash
   mkdir -p models
   ```

2. Download the models using the Hugging Face Hub:
   ```bash
   pip install huggingface-hub
   python download_models.py
   ```
   
   Or download manually and place in the `models` directory:
   - Base model: `Meta-Llama-3-8B-Q4_K_M.gguf`
   - Instruct model: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`

## Running the Application

1. Start the server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## Usage

1. Enter your system prompt and user input in the respective fields
2. Adjust generation parameters as needed
3. Click "Generate" to see both models' responses
4. Compare the outputs side by side

## API Endpoints

- `GET /` - Web interface
- `GET /api/models` - List available models
- `WEBSOCKET /ws` - WebSocket endpoint for streaming responses

## Project Structure

- `app.py` - Main FastAPI application
- `static/` - Frontend assets (HTML, CSS, JavaScript)
- `models/` - Directory for model files
- `download_models.py` - Script to download required models

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for efficient model inference
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Meta Llama](https://ai.meta.com/llama/) for the language models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

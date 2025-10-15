# Model Selection Feature

## What's New

The UI now dynamically loads all available Ollama models and lets you select which ones to compare!

## Your Current Models

```bash
$ ollama list
NAME                       ID              SIZE      MODIFIED       
llama3:text                870a5d02cfaf    4.7 GB    12 minutes ago    
llama3:instruct            365c0bd3c000    4.7 GB    14 minutes ago    
llama3-groq-tool-use:8b    36211dad2b15    4.7 GB    15 minutes ago
```

## How It Works

### 1. **Model Dropdowns**
- **Left Side**: Select any model (e.g., `llama3:text`)
- **Right Side**: Select any model (e.g., `llama3:instruct`)

### 2. **Dynamic Loading**
- On page load, fetches all models from Ollama
- Populates both dropdowns with available models
- Updates in real-time when you change selection

### 3. **Comparison Examples**

**Base vs Instruct:**
- Left: `llama3:text` (base model)
- Right: `llama3:instruct` (RLHF tuned)

**Instruct vs Tool Use:**
- Left: `llama3:instruct`
- Right: `llama3-groq-tool-use:8b`

**Any Two Models:**
- Compare any combination you want!

## Usage

### 1. Start the App
```bash
source venv/bin/activate
python app_ollama.py
```

### 2. Open Browser
http://localhost:8000

### 3. Select Models
- Use the dropdowns to pick which models to compare
- The pane headers show which model is active
- Click "Generate" to run both models side-by-side

## Adding More Models

```bash
# Pull any Ollama model
ollama pull mistral:7b
ollama pull gemma2:9b
ollama pull codellama:13b

# Refresh the page - new models appear automatically!
```

## API Endpoints

### GET /api/models
Returns list of available models:
```json
{
  "models": [
    "llama3:text",
    "llama3:instruct", 
    "llama3-groq-tool-use:8b"
  ],
  "current": {
    "base": "qwen2.5:7b-base",
    "instruct": "qwen2.5:7b"
  }
}
```

### GET /api/health
Returns Ollama connection status and configured models.

## Features

✅ **Dynamic model loading** - No code changes needed
✅ **Real-time selection** - Pick models from dropdowns
✅ **Visual feedback** - See which model is active
✅ **Flexible comparison** - Compare any two models
✅ **Auto-refresh** - Pull new models, refresh page

## Tips

1. **Compare base vs fine-tuned** to see RLHF effects
2. **Compare different fine-tuning approaches** (instruct vs tool-use)
3. **Test prompt sensitivity** across different models
4. **Evaluate model capabilities** side-by-side

Enjoy comparing your models! 🚀

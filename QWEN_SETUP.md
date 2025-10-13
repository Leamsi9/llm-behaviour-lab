# Qwen 2.5 Model Setup

Your app has been updated to use **Qwen 2.5 14B** models instead of Llama 3.

## Why Qwen 2.5?

- ✅ **Official GGUF files** from Qwen team (no vocabulary errors)
- ✅ **Reliable tokenizer** - no "byte not found in vocab" issues
- ✅ **Excellent quality** - competitive with Llama 3
- ✅ **Both base and instruct** versions available
- ✅ **Well-documented RLHF** training process

## Download Models

### Option 1: Automated Script (Recommended)

```bash
./download_qwen_models.sh
```

This will download both models (~18GB total) to the `models/` directory.

### Option 2: Manual Download

```bash
cd models

# Base model (~9GB)
wget https://huggingface.co/Qwen/Qwen2.5-14B-GGUF/resolve/main/qwen2.5-14b-q4_k_m.gguf \
     -O Qwen2.5-14B-Q4_K_M.gguf

# Instruct model (~9GB)
wget https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf \
     -O Qwen2.5-14B-Instruct-Q4_K_M.gguf
```

### Option 3: Smaller 7B Models (If 14B is too large)

If you want smaller/faster models:

```bash
cd models

# 7B Base (~4.5GB)
wget https://huggingface.co/Qwen/Qwen2.5-7B-GGUF/resolve/main/qwen2.5-7b-q4_k_m.gguf \
     -O Qwen2.5-7B-Q4_K_M.gguf

# 7B Instruct (~4.5GB)
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf \
     -O Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

Then update `app.py` lines 21-22:
```python
DEFAULT_MODEL = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
DEFAULT_BASE_MODEL = "Qwen2.5-7B-Q4_K_M.gguf"
```

## What Changed in the Code

1. **Model filenames** updated to Qwen 2.5
2. **Chat template** changed from Llama 3 format to Qwen format:
   - Llama 3: `<|start_header_id|>...<|eot_id|>`
   - Qwen: `<|im_start|>...<|im_end|>`
3. **Stop tokens** updated to `["<|im_end|>"]`
4. **MODEL_TYPE** variable added for easy switching

## Qwen vs Llama 3

| Feature | Qwen 2.5 14B | Llama 3 8B |
|---------|--------------|------------|
| **Parameters** | 14B | 8B |
| **Quality** | Excellent | Excellent |
| **GGUF Files** | Official ✅ | Third-party ⚠️ |
| **Vocab Issues** | None ✅ | Your files broken ❌ |
| **RAM (Q4_K_M)** | ~9GB each | ~5GB each |
| **Speed** | Slower (more params) | Faster |
| **Multilingual** | Better | Good |

## Switching Back to Llama 3

If you want to use Llama 3 later (with working GGUF files):

1. Download proper Llama 3 models from bartowski
2. Update `app.py` line 27: `MODEL_TYPE = "llama"`
3. Update model filenames in lines 21-22

## Testing

After downloading models:

1. Restart your app (it will auto-reload)
2. Check terminal for "✓ Successfully loaded" messages
3. Open http://localhost:8000
4. Try both base and instruct models

## Troubleshooting

**Models not loading?**
- Check file names match exactly (case-sensitive)
- Verify files are in `models/` directory
- Check terminal output for specific errors

**Out of memory?**
- Use 7B models instead of 14B
- Close other applications
- Consider Q3_K_M quantization (smaller but lower quality)

## Resources

- [Qwen 2.5 Official Repo](https://github.com/QwenLM/Qwen2.5)
- [Qwen 2.5 HuggingFace](https://huggingface.co/Qwen)
- [GGUF Format Documentation](https://github.com/ggerganov/llama.cpp/blob/master/docs/GGUF.md)

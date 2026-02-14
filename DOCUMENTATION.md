# LLMWS Enhanced - Complete Documentation

## Overview

LLMWS Enhanced is a WebSocket-based LLM server with advanced features:
- **Session persistence** - Reconnect without losing context
- **Model management** - Download, switch, and manage models
- **CUDA memory estimation** - Automatic context limit calculation
- **Simple encryption** - Session buffers protected with base64 XOR
- **Git-LFS integration** - Download models from Hugging Face
- **Resource monitoring** - Real-time GPU/memory stats
- **KISS principle** - Simple, clean, functional code

## Directory Structure

```
llmws/
├── llmws.py                    # Main server
├── llm.py                      # CLI client
├── web_client_enhanced.html    # Web interface
├── config.json                 # Server configuration
│
├── models/                     # Downloaded models
│   ├── Llama-3.1-Tulu-3-8B/
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── model.safetensors
│   └── Qwen2.5-VL-7B/
│       └── ...
│
└── var/                        # Runtime data
    ├── sessions/               # Session buffers (encrypted)
    │   ├── <session-uuid-1>/
    │   │   └── buffer.jsonl    # Encrypted response buffer
    │   └── <session-uuid-2>/
    │       └── buffer.jsonl
    │
    ├── models/                 # User-trained models
    │   └── my-finetuned-model/
    │       └── ...
    │
    └── logs/                   # Server logs
        └── llmws.log
```

## Configuration File (config.json)

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8765,
    "max_message_size": 16777216,
    "ping_interval": 30,
    "ping_timeout": 10
  },
  
  "model": {
    "path": null,              // Auto-detect if null
    "use_safetensors": true,
    "trust_remote_code": true,
    "device_map": "auto"
  },
  
  "optimization": {
    "use_flash_attention": true,
    "dtype": "auto",           // auto, bfloat16, float16
    "compile": false
  },
  
  "generation_defaults": {
    "max_new_tokens": 2048,
    "do_sample": true,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1
  },
  
  "paths": {
    "models_dir": "models",
    "var_dir": "var",
    "sessions_dir": "var/sessions",
    "logs_dir": "var/logs"
  },
  
  "limits": {
    "max_clients": 10,
    "queue_size_per_client": 1024,
    "max_context_length": 128000,
    "timeout_token_generation": 300
  }
}
```

## Protocol Specification

### Message Types

#### 1. Client → Server: Hello (First Message)

```json
{
  "session_id": "optional-uuid-to-reconnect"
}
```

#### 2. Server → Client: Welcome

```json
{
  "type": "welcome",
  "session_id": "generated-or-resumed-uuid",
  "model": "Llama-3.1-Tulu-3-8B",
  "capabilities": {
    "vision": false,
    "flash_attention": true,
    "compute_capability": [8, 9],
    "max_context": 32768
  },
  "resources": {
    "device": "NVIDIA RTX 4090",
    "total_gb": 24.0,
    "free_gb": 18.5,
    "compute_capability": [8, 9]
  }
}
```

#### 3. Client → Server: Inference Request

```json
{
  "type": "inference",
  "prompt": {
    "system": "You are a helpful assistant.",
    "user": "What is the capital of France?"
  },
  "config": {
    "temperature": 0.7,
    "max_new_tokens": 2048
  }
}
```

#### 4. Server → Client: Token Stream

```json
{"type": "start", "tokens_in": 45, "max_tokens": 2048}
{"type": "token", "data": "The "}
{"type": "token", "data": "capital "}
{"type": "token", "data": "of "}
{"type": "token", "data": "France "}
{"type": "token", "data": "is "}
{"type": "token", "data": "Paris."}
{"type": "done", "total_tokens": 12}
```

#### 5. Client → Server: Update Config

```json
{
  "type": "config",
  "data": {
    "temperature": 0.9,
    "max_new_tokens": 1024
  }
}
```

Response:
```json
{
  "type": "config_ack",
  "config": {
    "temperature": 0.9,
    "max_new_tokens": 1024,
    "top_p": 0.95,
    "top_k": 50
  }
}
```

#### 6. Client → Server: Acknowledge Response

```json
{
  "type": "ack"
}
```

Response:
```json
{
  "type": "ack_received"
}
```

**Effect**: Clears session buffer and removes session data.

#### 7. Client → Server: Get Resources

```json
{
  "type": "get_resources"
}
```

Response:
```json
{
  "type": "resources",
  "cuda": {
    "device": "NVIDIA RTX 4090",
    "total_gb": 24.0,
    "free_gb": 18.5,
    "allocated_gb": 5.2,
    "compute_capability": [8, 9]
  },
  "model": {
    "name": "Llama-3.1-Tulu-3-8B",
    "path": "/path/to/model",
    "vision": false
  },
  "available_models": [
    {
      "name": "Llama-3.1-Tulu-3-8B",
      "path": "models/Llama-3.1-Tulu-3-8B",
      "source": "models",
      "size_mb": 15360
    }
  ]
}
```

#### 8. Client → Server: List Models

```json
{
  "type": "list_models"
}
```

Response:
```json
{
  "type": "models_list",
  "models": [
    {
      "name": "Llama-3.1-Tulu-3-8B",
      "path": "models/Llama-3.1-Tulu-3-8B",
      "source": "models",
      "size_mb": 15360
    },
    {
      "name": "my-finetuned-model",
      "path": "var/models/my-finetuned-model",
      "source": "var",
      "size_mb": 8192
    }
  ]
}
```

#### 9. Client → Server: Switch Model

```json
{
  "type": "switch_model",
  "model_path": "models/Qwen2.5-VL-7B"
}
```

Response:
```json
{"type": "log", "message": "Loading model..."}
{"type": "log", "message": "✓ Model loaded"}
{"type": "model_switched", "message": "Model switched to ..."}
```

#### 10. Client → Server: Download Model (Git-LFS)

```json
{
  "type": "download_model",
  "repo": "https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B",
  "target_dir": "Llama-3.1-Tulu-3-8B"
}
```

Response (streaming):
```json
{"type": "download_started", "message": "Download started in background"}
{"type": "log", "message": "Starting download..."}
{"type": "log", "message": "Cloning repository..."}
{"type": "log", "message": "Downloading LFS objects..."}
{"type": "log", "message": "Download complete!"}
```

#### 11. Client → Server: Delete Model

```json
{
  "type": "delete_model",
  "model_name": "my-old-model"
}
```

Response:
```json
{
  "type": "model_deleted",
  "message": "Model my-old-model deleted"
}
```

#### 12. Server → Client: Error

```json
{
  "type": "error",
  "message": "CUDA out of memory. Context too long. Please retry with shorter input.",
  "max_context": 32768,
  "input_length": 45000
}
```

## Session Management

### Session Lifecycle

1. **Connect**: Client connects and sends optional `session_id`
2. **Welcome**: Server creates/resumes session and sends welcome
3. **Inference**: Client sends requests, server buffers responses
4. **Disconnect**: Connection lost (network issue, etc.)
5. **Reconnect**: Client reconnects with `session_id`
6. **Resume**: Server sends buffered responses
7. **Acknowledge**: Client sends `ack`, server clears buffer

### Session Buffer

- Stored in `var/sessions/<session-id>/buffer.jsonl`
- Each line is encrypted with simple XOR + base64
- Format: `{"id": "uuid", "data": "base64-encrypted-json"}`
- Cleared on:
  - Client sends `ack`
  - Server restart
  
### Simple Encryption

Not cryptographically secure, just obfuscation:

```python
def simple_encrypt(data: str) -> str:
    key = b"llmws"
    data_bytes = data.encode('utf-8')
    encrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(data_bytes))
    return base64.b64encode(encrypted).decode('ascii')
```

Already base64-encoded data (like images) is NOT re-encrypted.

## CUDA Memory Management

### Context Length Estimation

Server estimates maximum context based on available VRAM:

```python
def estimate_max_context(available_memory_gb, model_params_b):
    usable_memory_gb = available_memory_gb * 0.6  # Reserve for activations
    usable_memory_bytes = usable_memory_gb * 1024**3
    bytes_per_token = model_params_b * 1e9 * 2 / 1000
    max_tokens = int(usable_memory_bytes / bytes_per_token)
    return min(max_tokens, MAX_CONTEXT_LENGTH)
```

### Out of Memory Handling

If context exceeds estimated maximum:

```json
{
  "type": "error",
  "message": "Context too long (45000 tokens). Max: 32768. Please retry with shorter input.",
  "max_context": 32768,
  "input_length": 45000
}
```

Client should retry with shorter input.

If CUDA OOM during generation:

```json
{
  "type": "error",
  "message": "CUDA out of memory. Context too long. Please retry with shorter input."
}
```

Server clears CUDA cache automatically.

## Model Management

### Auto-Selection

On startup, if `config.model.path` is `null`:
1. Scan `models/` and `var/models/`
2. If only one model found → auto-select
3. If multiple found → prefer from `models/`, else first

### Downloading Models

Using git-lfs (must be installed):

```bash
# Install git-lfs first
sudo apt-get install git-lfs
git lfs install
```

Then use client:

```python
await client.download_model(
    repo="https://huggingface.co/org/model-name",
    target_dir="model-name"
)
```

Server runs download in background and streams logs.

### Switching Models

```python
await client.switch_model("models/Qwen2.5-VL-7B")
```

Server:
1. Unloads current model
2. Clears CUDA cache
3. Loads new model
4. Streams loading logs

### Deleting Models

Only models in `var/models/` can be deleted (safety):

```python
await client.delete_model("my-old-model")
```

Models in `models/` must be deleted manually.

## CLI Client Usage

### Basic Usage

```bash
# Interactive mode
python llm.py

# Single prompt
python llm.py --prompt "Hello"

# With config
python llm.py \
  --prompt "Write a haiku" \
  --temperature 0.9 \
  --max-tokens 100

# Reconnect to session
python llm.py \
  --session abc12345-...
```

### Interactive Commands

```
/config temperature=0.8       - Update config
/system You are helpful       - Set system prompt
/resources                    - Show server resources
/models                       - List available models
/switch models/Llama-3.1...   - Switch model
/download <repo> <dir>        - Download model
/delete <name>                - Delete model from var/
/session                      - Show current session ID
/quit                         - Exit
```

## Web Client Usage

1. Open `web_client_enhanced.html` in browser
2. Enter server address (default: `localhost:8765`)
3. Optional: Enter session ID to reconnect
4. Click "Connect"
5. Chat with the model

Features:
- Real-time token streaming
- Config adjustment
- Resource monitoring
- Model management
- Download models via git-lfs

## Python API

```python
from llm import LLMWSEnhancedClient

async def main():
    # Connect
    client = LLMWSEnhancedClient(host="localhost", port=8765)
    await client.connect()
    
    # Configure
    await client.set_config(
        temperature=0.8,
        max_new_tokens=1024
    )
    
    # Inference
    response = await client.inference(
        prompt="What is AI?",
        system="You are a helpful assistant"
    )
    
    # Get resources
    await client.get_resources()
    
    # List models
    models = await client.list_models()
    
    # Switch model
    await client.switch_model("models/Qwen2.5-VL-7B")
    
    # Download model
    await client.download_model(
        repo="https://huggingface.co/org/model",
        target_dir="model-name"
    )
    
    # Disconnect (with ack to clear buffer)
    await client.disconnect(send_ack=True)

asyncio.run(main())
```

## Request Queue

Server processes requests in FIFO order:
- Multiple clients can connect
- Each client has a queue
- Server processes one request at a time
- Clients wait for their turn
- Can disconnect/reconnect without losing position

## Backward Compatibility

The enhanced server maintains backward compatibility with the original LLMWS:

### Legacy Format Still Works

```python
# Old style prompt
message = "<|system|>You are helpful<|end|><|user|>Hello<|end|>"

# Server parses both formats
```

### Legacy Message Types

```json
{
  "type": "inference",
  "prompt": "<|system|>...<|end|><|user|>...<|end|>"
}
```

Works alongside new format:

```json
{
  "type": "inference",
  "prompt": {
    "system": "...",
    "user": "..."
  }
}
```

## Performance Tuning

### For RTX 4090 / Ada (SM 8.9)

```json
{
  "optimization": {
    "use_flash_attention": true,
    "dtype": "bfloat16"
  }
}
```

### For RTX 3090 / Ampere (SM 8.6)

```json
{
  "optimization": {
    "use_flash_attention": true,
    "dtype": "float16"
  }
}
```

### For Blackwell / GB200 (SM 10.0)

```json
{
  "optimization": {
    "use_flash_attention": true,
    "dtype": "bfloat16"
  }
}
```

Server automatically optimizes for detected architecture.

## Error Handling

### Connection Errors

```python
try:
    await client.connect()
except Exception as e:
    print(f"Connection failed: {e}")
```

### Inference Errors

```json
{
  "type": "error",
  "message": "Error description"
}
```

Common errors:
- `"CUDA out of memory"` - Context too long
- `"Context too long (...)"` - Exceeds estimated max
- `"Model not found"` - Invalid model path
- `"Queue full"` - Too many pending requests

### Recovery

1. CUDA OOM → Retry with shorter input
2. Connection lost → Reconnect with session_id
3. Model switch failed → Check logs, try again

## Security Notes

### Session Encryption

- Simple XOR + base64 (NOT cryptographically secure)
- Purpose: Obfuscation, not security
- Already-encoded data (base64 images) not re-encrypted
- Session buffers cleared on restart

### Network Security

- No authentication by default
- Use firewall/VPN for production
- Consider adding TLS/SSL for production
- Don't expose to public internet without protection

### Model Safety

- Only models in `var/models/` can be deleted via API
- Models in `models/` require manual deletion
- Downloaded models verified by git-lfs

## Troubleshooting

### Server won't start

1. Check `config.json` is valid JSON
2. Ensure directories exist: `models/`, `var/`
3. Check port 8765 is available

### Client can't connect

1. Server running?
2. Correct host:port?
3. Firewall blocking?

### Out of memory errors

1. Check available VRAM: `/resources`
2. Reduce max_new_tokens
3. Use shorter prompts
4. Switch to smaller model

### Model download fails

1. Is git-lfs installed?
2. Check network connection
3. Verify repo URL
4. Check disk space

### Session reconnect fails

1. Session may have expired (server restart)
2. Check session_id is correct
3. Buffers cleared on server restart

## Architecture Principles

### KISS (Keep It Simple, Stupid)

Following Arch Linux philosophy:
- Minimal abstractions
- Clear code flow
- No unnecessary complexity
- One way to do things
- Easy to understand and modify

### Key Design Decisions

1. **Single-threaded processing** - FIFO queue, no concurrency issues
2. **Simple encryption** - XOR obfuscation, not crypto
3. **File-based sessions** - JSONL, easy to debug
4. **Auto-cleanup** - Sessions cleared on restart
5. **Backward compatible** - Legacy formats still work

## File Descriptions

### Core Files

- `llmws_server_enhanced.py` (800 lines)
  - Main server logic
  - Model loading/management
  - Session handling
  - CUDA memory estimation
  
- `llmws_client_enhanced.py` (400 lines)
  - CLI client
  - Session management
  - All API methods
  
- `web_client_enhanced.html` (450 lines)
  - Modern web interface
  - Real-time streaming
  - Model management UI
  
- `config.json`
  - Server configuration
  - Auto-generated on first run

### Data Directories

- `models/` - Downloaded models (git-lfs)
- `var/sessions/` - Session buffers (encrypted JSONL)
- `var/models/` - User-trained models
- `var/logs/` - Server logs

## Maintenance

### Session Cleanup

Sessions are automatically cleared on:
- Server restart
- Client sends `ack`

Manual cleanup:
```bash
rm -rf var/sessions/*
```

### Model Cleanup

```bash
# List models
du -sh models/*
du -sh var/models/*

# Delete manually
rm -rf var/models/old-model
```

### Logs

```bash
# View logs
tail -f var/logs/llmws.log

# Rotate logs
mv var/logs/llmws.log var/logs/llmws.log.old
```

## Examples

See original documentation for examples:
- Text generation
- Code generation
- Vision models (if supported)
- Multi-turn conversations

Plus new features:
- Session reconnect
- Model switching
- Resource monitoring
- Model downloads

---

**LLMWS Enhanced** - Simple, powerful, reliable WebSocket LLM server.

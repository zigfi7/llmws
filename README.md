# LLMWS

WebSocket LLM Server with session persistence, model management, and CUDA optimization.

## Features

✅ **Session Persistence** - Reconnect without losing context  
✅ **Model Management** - Download, switch, delete models via API  
✅ **CUDA Memory Estimation** - Automatic context limit calculation  
✅ **Git-LFS Integration** - Download models from Hugging Face  
✅ **Resource Monitoring** - Real-time GPU/VRAM stats  
✅ **Simple Encryption** - Session buffers protected  
✅ **Backward Compatible** - Works with original LLMWS clients  
✅ **KISS Principle** - Clean, simple, maintainable code  

## Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers safetensors websockets pillow --break-system-packages
```

### 2. Download a Model

```bash
mkdir -p models
cd models
git lfs install
git clone https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B
cd ..
```

### 3. Start Server

```bash
python llmws.py
```

Output:
```
============================================================
LLMWS Enhanced Server
============================================================

Available models: 1
  - Llama-3.1-Tulu-3-8B (15360 MB) [models]

Loading model from: models/Llama-3.1-Tulu-3-8B
✓ Model loaded: Llama-3.1-Tulu-3-8B
✓ Tokenizer loaded

============================================================
Server ready on ws://0.0.0.0:8765
============================================================
```

### 4. Use Client

**CLI:**
```bash
python llm.py
```

**Web:**
Open `web_client_enhanced.html` in browser.

## Directory Structure

```
llmws/
├── llmws.py                          # Server
├── llm.py                            # CLI client
├── web_client_enhanced.html          # Web client
├── config.json                       # Configuration
├── DOCUMENTATION.md                  # Full docs
│
├── models/                           # Downloaded models
│   └── Llama-3.1-Tulu-3-8B/
│
└── var/                              # Runtime data
    ├── sessions/                     # Session buffers
    ├── models/                       # Trained models
    └── logs/                         # Logs
```

## Usage

### CLI Client

```bash
# Interactive mode
python llm.py

# Single prompt
python llm.py --prompt "Hello"

# Reconnect to session
python llm.py --session abc12345-...
```

### Interactive Commands

```
/config temperature=0.8    - Update config
/resources                 - Show GPU/VRAM stats
/models                    - List available models
/switch models/Qwen...     - Switch model
/download <repo> <dir>     - Download model
/session                   - Show session ID
/quit                      - Exit
```

### Web Client

1. Open `web_client_enhanced.html`
2. Enter server: `localhost:8765`
3. Click "Connect"
4. Start chatting

## Configuration

Edit `config.json`:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8765
  },
  "model": {
    "path": null  // Auto-detect
  },
  "generation_defaults": {
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50
  }
}
```

## Python API

```python
from llm import LLMWSEnhancedClient

async def main():
    client = LLMWSEnhancedClient()
    await client.connect()
    
    # Chat
    response = await client.inference(
        prompt="What is AI?",
        system="You are helpful"
    )
    
    # Get resources
    await client.get_resources()
    
    # Switch model
    await client.switch_model("models/Qwen2.5-VL-7B")
    
    await client.disconnect()

asyncio.run(main())
```

## Protocol

All messages are JSON over WebSocket.

### Inference

**Request:**
```json
{
  "type": "inference",
  "prompt": {
    "system": "You are helpful",
    "user": "Hello"
  }
}
```

**Response (streaming):**
```json
{"type": "start", "tokens_in": 23, "max_tokens": 2048}
{"type": "token", "data": "Hello"}
{"type": "token", "data": "! How"}
{"type": "token", "data": " can I help?"}
{"type": "done", "total_tokens": 8}
```

### Session Management

**Connect with session:**
```json
{"session_id": "uuid-to-resume"}
```

**Acknowledge (clears buffer):**
```json
{"type": "ack"}
```

### Model Management

**List models:**
```json
{"type": "list_models"}
```

**Switch model:**
```json
{
  "type": "switch_model",
  "model_path": "models/Llama-3.1-Tulu-3-8B"
}
```

**Download model:**
```json
{
  "type": "download_model",
  "repo": "https://huggingface.co/org/model",
  "target_dir": "model-name"
}
```

**Get resources:**
```json
{"type": "get_resources"}
```

## Features in Detail

### Session Persistence

- Each client gets unique session ID
- Responses buffered in `var/sessions/<uuid>/buffer.jsonl`
- Simple XOR+base64 encryption
- Reconnect with session_id to resume
- Send `ack` to clear buffer
- Auto-cleanup on server restart

### CUDA Memory Management

Server estimates max context based on available VRAM:

```
Usable Memory = Free VRAM × 0.6
Bytes per Token = Model Params × 2 / 1000
Max Context = Usable Memory / Bytes per Token
```

If context exceeds limit:
```json
{
  "type": "error",
  "message": "Context too long (45000 tokens). Max: 32768. Please retry.",
  "max_context": 32768,
  "input_length": 45000
}
```

### Model Auto-Selection

On startup:
1. Scan `models/` and `var/models/`
2. If 1 model → auto-select
3. If multiple → prefer from `models/`
4. If none → wait for client to download

### Git-LFS Downloads

```bash
# Install git-lfs first
sudo apt-get install git-lfs
git lfs install
```

Then use client:
```python
await client.download_model(
    repo="https://huggingface.co/org/model",
    target_dir="model-name"
)
```

Server runs download in background, streams logs.

## KISS Architecture

Following Arch Linux principles:
- **Simple** - Clear code flow
- **Functional** - One job, done well
- **Maintainable** - Easy to understand
- **No magic** - Explicit, not implicit
- **Minimal** - No unnecessary features

## Backward Compatibility

Works with original LLMWS:
- Legacy prompt format: `<|system|>...<|end|><|user|>...<|end|>`
- Same token streaming
- Same WebSocket protocol
- New features are additive

## Performance

- **Flash Attention 2** - Automatic on SM 8.0+
- **Safetensors** - Fast loading
- **BFloat16** - On Ampere+ (SM 8.0+)
- **CUDA 13+** - Optimized for latest GPUs
- **Blackwell** - SM 10.0 support

## Troubleshooting

### Out of Memory

```bash
# Check resources
python llm.py
> /resources

# Reduce tokens
> /config max_new_tokens=512
```

### Model Download Fails

```bash
# Check git-lfs
git lfs version

# Install if missing
sudo apt-get install git-lfs
git lfs install
```

### Connection Refused

```bash
# Check server is running
ps aux | grep llmws

# Check port
netstat -tuln | grep 8765
```

## Documentation

- **DOCUMENTATION.md** - Complete reference
- **README.md** (this file) - Quick start
- **config.json** - Configuration reference

## Examples

### Basic Chat
```python
client = LLMWSEnhancedClient()
await client.connect()
await client.inference("What is quantum computing?")
await client.disconnect()
```

### Multi-turn with Session
```python
# Session 1
client = LLMWSEnhancedClient()
await client.connect()
session_id = client.session_id
await client.inference("My name is Alice")
# Network disconnects...

# Session 2 - Resume
client = LLMWSEnhancedClient()
await client.connect(session_id=session_id)
await client.inference("What's my name?")
# Response: "Your name is Alice"
```

### Model Management
```python
client = LLMWSEnhancedClient()
await client.connect()

# List models
models = await client.list_models()

# Switch
await client.switch_model("models/Qwen2.5-VL-7B")

# Download new model
await client.download_model(
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B",
    "Qwen2.5-VL-7B"
)
```

## Security Notes

⚠️ **Session encryption is NOT cryptographically secure**
- Simple XOR + base64 obfuscation
- Don't store sensitive data in sessions
- Use VPN/firewall for production
- Consider TLS/SSL for production deployments

## Contributing

Keep it KISS:
- Clear code
- No unnecessary abstractions
- Comments for non-obvious logic
- Test before committing

## License

Same as original LLMWS project.

---

**Simple. Powerful. Reliable.**

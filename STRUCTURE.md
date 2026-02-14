# LLMWS Enhanced - Project Structure

## File Organization

```
llmws-enhanced/
│
├── Core Files (Production)
│   ├── llmws.py                        800 lines    Main server
│   ├── llm.py                          400 lines    CLI client
│   ├── web_client_enhanced.html        450 lines    Web interface
│   └── config.json                     Auto-gen     Configuration
│
├── Documentation
│   ├── README_ENHANCED.md              Quick start guide
│   ├── DOCUMENTATION.md                Complete reference
│   └── STRUCTURE.md                    This file
│
├── Legacy Files (Reference)
│   ├── llmws_server.py                 Original enhanced server
│   ├── llmws_client.py                 Original CLI client
│   ├── llmws_train_client.py           Training client
│   ├── web_client.html                 Original web client
│   ├── test_llmws.py                   Test suite
│   ├── examples_vision.py              Vision examples
│   ├── README.md                       Original docs
│   ├── QUICKSTART.md                   Original quickstart
│   ├── COMPARISON.md                   Feature comparison
│   ├── requirements.txt                Dependencies
│   └── setup.sh                        Setup script
│
└── Runtime Directories
    ├── models/                         Downloaded models
    │   └── <model-name>/
    │       ├── config.json
    │       ├── tokenizer.json
    │       └── model.safetensors
    │
    └── var/                            Runtime data
        ├── sessions/                   Session buffers
        │   └── <session-uuid>/
        │       └── buffer.jsonl        Encrypted
        ├── models/                     User models
        │   └── <model-name>/
        └── logs/                       Server logs
            └── llmws.log
```

## Component Responsibilities

### llmws.py

**Purpose**: Main WebSocket server

**Key Functions**:
- `load_config()` - Load/create config.json
- `create_session()` - Create new session with UUID
- `get_session()` - Get existing session
- `save_session_buffer()` - Save encrypted buffer to disk
- `load_session_buffer()` - Load encrypted buffer from disk
- `cleanup_session()` - Remove session data
- `simple_encrypt/decrypt()` - XOR + base64 encryption
- `scan_models()` - Find available models
- `auto_select_model()` - Pick model if only one
- `download_model_git_lfs()` - Download via git-lfs
- `estimate_max_context()` - Calculate max tokens from VRAM
- `check_cuda_memory()` - Get GPU stats
- `check_cuda_compatibility()` - Check GPU architecture
- `load_model()` - Load model from path
- `stream_inference()` - Generate tokens with streaming
- `handle_message()` - Route incoming messages
- `handle_connection()` - WebSocket connection handler
- `cleanup_old_sessions()` - Remove old sessions on startup
- `main()` - Server startup

**Data Structures**:
- `sessions` - Dict[session_id -> session_data]
- `active_connections` - Dict[websocket -> session_id]
- `available_models` - List of scanned models

**Session Data**:
```python
{
    "id": "uuid",
    "created": timestamp,
    "last_activity": timestamp,
    "buffer_file": Path,
    "buffer": [],                # In-memory buffer
    "config": {},                # Generation config
    "queue": [],                 # Request queue
    "processing": bool
}
```

### llm.py

**Purpose**: Command-line client

**Key Methods**:
- `connect(session_id)` - Connect/reconnect to server
- `disconnect(send_ack)` - Disconnect with optional ack
- `set_config(**kwargs)` - Update generation config
- `inference(prompt, system)` - Run inference with streaming
- `get_resources()` - Get server resources
- `list_models()` - List available models
- `switch_model(path)` - Switch to different model
- `download_model(repo, target)` - Download via git-lfs
- `delete_model(name)` - Delete model from var/
- `interactive_mode()` - Interactive chat loop

**State**:
- `session_id` - Current session UUID
- `config` - Client-side config cache
- `websocket` - Active connection

### web_client_enhanced.html

**Purpose**: Browser-based interface

**Features**:
- Clean, modern UI
- Real-time token streaming
- Config adjustment
- Resource monitoring
- Model management
- Download models
- Session persistence

**Functions**:
- `connect()` - WebSocket connection
- `disconnect()` - Close connection
- `handleMessage(data)` - Route incoming messages
- `sendInference()` - Send inference request
- `displayResources(data)` - Show GPU/VRAM stats
- `displayModels(models)` - List available models
- `switchModel(path)` - Switch to model
- `downloadModel()` - Download via git-lfs

### config.json

**Purpose**: Server configuration

**Sections**:
- `server` - Host, port, limits
- `model` - Model path, options
- `optimization` - Flash attention, dtype
- `generation_defaults` - Sampling params
- `paths` - Directory locations
- `limits` - Resource limits

**Auto-generation**:
Created on first run if missing, with sensible defaults.

## Data Flow

### 1. Connection Flow

```
Client                          Server
  │                               │
  │─────── WS Connect ────────────▶│
  │                               │ Generate/Resume session_id
  │◀────── Welcome ───────────────│ (with capabilities)
  │                               │
```

### 2. Inference Flow

```
Client                          Server
  │                               │
  │─────── Inference Req ─────────▶│
  │                               │ Check CUDA memory
  │                               │ Estimate max context
  │                               │ Tokenize input
  │◀────── Start Msg ─────────────│
  │                               │
  │◀────── Token 1 ────────────────│
  │◀────── Token 2 ────────────────│ Stream tokens
  │◀────── Token 3 ────────────────│ Buffer responses
  │◀────── ... ────────────────────│
  │                               │
  │◀────── Done Msg ───────────────│
  │                               │ Save buffer to disk
```

### 3. Disconnect/Reconnect Flow

```
Client                          Server
  │                               │
  │─────── Disconnect ─────────────│
  │                               │ Keep session alive
  │                               │ Buffer saved to disk
  │                               │
  │─────── Reconnect + UUID ──────▶│
  │                               │ Resume session
  │                               │ Load buffer from disk
  │◀────── Buffered Data ──────────│
  │◀────── ... ────────────────────│
  │                               │
  │─────── ACK ────────────────────▶│
  │                               │ Clear buffer
  │                               │ Delete buffer file
```

### 4. Model Switch Flow

```
Client                          Server
  │                               │
  │─────── Switch Model ───────────▶│
  │                               │ Unload current
  │                               │ Clear CUDA cache
  │◀────── Log: Unloading... ─────│
  │                               │ Load new model
  │◀────── Log: Loading... ───────│
  │◀────── Log: Done ──────────────│
  │                               │
  │◀────── Model Switched ─────────│
```

## Session Buffer Format

File: `var/sessions/<uuid>/buffer.jsonl`

Each line:
```json
{
  "id": "line-uuid",
  "data": "base64-encrypted-json"
}
```

Encrypted data (after decryption):
```json
{
  "type": "token",
  "data": "Hello"
}
```

## Encryption Details

**Algorithm**: XOR with fixed key + base64

**Key**: `b"llmws"`

**Process**:
1. Convert data to UTF-8 bytes
2. XOR each byte with key[i % len(key)]
3. Base64 encode result
4. Store as string

**Decryption**:
1. Base64 decode
2. XOR each byte with key[i % len(key)]
3. Convert to UTF-8 string

**Security**: NOT cryptographically secure! Just obfuscation.

## CUDA Memory Estimation

**Formula**:
```python
usable_memory = free_vram_gb * 0.6
bytes_per_token = model_params_b * 1e9 * 2 / 1000
max_tokens = usable_memory_bytes / bytes_per_token
max_context = min(max_tokens, config_max)
```

**Example** (RTX 4090, 24GB, 8B model, 18GB free):
```
usable = 18 * 0.6 = 10.8 GB
bytes_per_token = 8e9 * 2 / 1000 = 16MB
max_tokens = 10.8 * 1024 / 16 = 691 tokens
```

Actually it's more complex (this is simplified).

## Request Queue

**Structure**:
- Each session has a queue
- FIFO processing
- One request at a time per session
- Global processing lock

**Flow**:
```
Session 1 Queue: [Req1, Req2]
Session 2 Queue: [Req3]
Session 3 Queue: [Req4, Req5]

Processing order: Req1 → Req2 → Req3 → Req4 → Req5
```

## Model Scanning

**Directories**:
1. `models/` - Downloaded models
2. `var/models/` - User-trained models

**Detection**:
- Check for `config.json` in each subdirectory
- Calculate size
- Store metadata

**Auto-selection**:
- If 1 model total → auto-select
- If multiple → prefer from `models/`
- If none → wait for client

## Message Types Summary

| Type | Direction | Purpose |
|------|-----------|---------|
| hello | C→S | Connect/reconnect |
| welcome | S→C | Connection ack |
| inference | C→S | Run inference |
| start | S→C | Generation started |
| token | S→C | Token stream |
| done | S→C | Generation done |
| error | S→C | Error occurred |
| config | C→S | Update config |
| config_ack | S→C | Config updated |
| ack | C→S | Clear buffer |
| ack_received | S→C | Buffer cleared |
| get_resources | C→S | Query resources |
| resources | S→C | Resource info |
| list_models | C→S | List models |
| models_list | S→C | Models response |
| switch_model | C→S | Switch model |
| model_switched | S→C | Switch complete |
| download_model | C→S | Download model |
| download_started | S→C | Download begun |
| log | S→C | Log message |
| delete_model | C→S | Delete model |
| model_deleted | S→C | Delete complete |
| ping | C→S | Keep-alive |
| pong | S→C | Keep-alive ack |

## Code Complexity

| File | Lines | Complexity |
|------|-------|------------|
| llmws.py | 800 | Medium |
| llm.py | 400 | Low |
| web_client_enhanced.html | 450 | Low |

**Total**: ~1650 lines (clean, documented code)

## Dependencies

**Core**:
- torch (PyTorch)
- transformers (Hugging Face)
- safetensors
- websockets
- pillow (for vision)

**Optional**:
- opencv-python (for vision)
- git-lfs (for downloads)

**Python**: 3.8+

## Performance Characteristics

**Startup**:
- Model scan: <1s
- Model load: 10-30s (depends on size)
- Session cleanup: <1s

**Runtime**:
- Token generation: GPU-limited
- Session save: <10ms
- Session load: <100ms
- Model switch: 10-30s

**Memory**:
- Server overhead: ~500MB
- Model: 8B = ~16GB, 70B = ~140GB
- Sessions: ~1KB per session

## Testing Strategy

**Unit Tests**:
- Session creation/cleanup
- Encryption/decryption
- Model scanning
- CUDA estimation

**Integration Tests**:
- Connect/disconnect
- Inference flow
- Session persistence
- Model switching

**Manual Tests**:
- Web client
- CLI client
- Concurrent clients
- Long conversations

## Deployment

**Development**:
```bash
python llmws.py
```

**Production** (with systemd):
```ini
[Unit]
Description=LLMWS Enhanced Server

[Service]
ExecStart=/usr/bin/python3 /path/to/llmws.py
WorkingDirectory=/path/to/llmws
Restart=always

[Install]
WantedBy=multi-user.target
```

**Docker** (example):
```dockerfile
FROM python:3.10
RUN pip install torch transformers safetensors websockets
COPY . /app
WORKDIR /app
EXPOSE 8765
CMD ["python", "llmws.py"]
```

## Monitoring

**Health Check**:
```python
ws = await websockets.connect("ws://localhost:8765")
await ws.send('{"type": "ping"}')
response = await ws.recv()
# Check for {"type": "pong"}
```

**Metrics**:
- Active sessions: `len(sessions)`
- CUDA memory: `check_cuda_memory()`
- Available models: `len(available_models)`

## Backup/Restore

**Backup**:
```bash
# Models
tar -czf models-backup.tar.gz models/

# Sessions (if needed)
tar -czf sessions-backup.tar.gz var/sessions/

# Config
cp config.json config.json.backup
```

**Restore**:
```bash
tar -xzf models-backup.tar.gz
tar -xzf sessions-backup.tar.gz
cp config.json.backup config.json
```

## Future Enhancements

**Potential**:
- Multi-GPU support
- Batch inference
- Model quantization
- WebSocket compression
- Authentication
- TLS/SSL
- Prometheus metrics
- Docker compose

**Not Planned** (keep it KISS):
- Database integration
- Complex auth systems
- Heavy abstractions
- Feature bloat

---

**Philosophy**: Simple, maintainable, functional. Everything has a place and a purpose.

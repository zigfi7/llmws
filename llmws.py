#!/usr/bin/env python3
"""
LLMWS Enhanced Server - WebSocket LLM Server
Features: Session persistence, model management, CUDA estimation, git-lfs support
KISS principle: Simple, clean, functional
"""
import os, sys, re, asyncio, json, base64, uuid, zlib, subprocess, shutil, time
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field

import torch
import websockets
from websockets.exceptions import WebSocketException
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoProcessor
from transformers.utils import is_flash_attn_2_available
from safetensors.torch import save_file
from PIL import Image
from io import BytesIO

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "server": {
        "host": "0.0.0.0",
        "port": 8765,
        "max_message_size": 16777216,
        "ping_interval": 30,
        "ping_timeout": 10
    },
    "model": {
        "path": None,  # Auto-detect from models/
        "use_safetensors": True,
        "trust_remote_code": True,
        "device_map": "auto"
    },
    "optimization": {
        "use_flash_attention": True,
        "dtype": "auto",
        "compile": False
    },
    "generation_defaults": {
        "max_new_tokens": 2048,
        "do_sample": True,
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

def load_config():
    """Load or create config file"""
    if Path(CONFIG_FILE).exists():
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        # Merge with defaults
        for key in DEFAULT_CONFIG:
            if key not in config:
                config[key] = DEFAULT_CONFIG[key]
            elif isinstance(DEFAULT_CONFIG[key], dict):
                for subkey in DEFAULT_CONFIG[key]:
                    if subkey not in config[key]:
                        config[key][subkey] = DEFAULT_CONFIG[key][subkey]
        return config
    else:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG

CONFIG = load_config()

# Create directories
for dir_key in ['models_dir', 'var_dir', 'sessions_dir', 'logs_dir']:
    os.makedirs(CONFIG['paths'][dir_key], exist_ok=True)

# ============================================================================
# GLOBALS
# ============================================================================

model = None
tokenizer = None
processor = None
generation_config = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
compute_capability = (0, 0)
use_flash_attention = False
available_models = []  # List of available models

# Session management
sessions: Dict[str, Dict] = {}  # session_id -> session data
active_connections: Dict[Any, str] = {}  # websocket -> session_id
request_queue: asyncio.Queue = asyncio.Queue()
processing_lock = asyncio.Lock()
current_model = None

# ============================================================================
# SIMPLE BASE64 ENCRYPTION
# ============================================================================

def simple_encrypt(data: str) -> str:
    """Simple base64 encryption (not secure, just obfuscation)"""
    if not data:
        return ""
    # XOR with a simple key, then base64
    key = b"llmws"
    data_bytes = data.encode('utf-8')
    encrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(data_bytes))
    return base64.b64encode(encrypted).decode('ascii')

def simple_decrypt(encrypted: str) -> str:
    """Simple base64 decryption"""
    if not encrypted:
        return ""
    try:
        encrypted_bytes = base64.b64decode(encrypted)
        key = b"llmws"
        decrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(encrypted_bytes))
        return decrypted.decode('utf-8')
    except:
        return ""

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def create_session(session_id: str = None) -> str:
    """Create new session"""
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    session_dir = Path(CONFIG['paths']['sessions_dir']) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    sessions[session_id] = {
        "id": session_id,
        "created": time.time(),
        "last_activity": time.time(),
        "buffer_file": session_dir / "buffer.jsonl",
        "buffer": [],
        "config": CONFIG['generation_defaults'].copy(),
        "queue": [],
        "processing": False
    }
    
    return session_id

def get_session(session_id: str) -> Optional[Dict]:
    """Get existing session or None"""
    return sessions.get(session_id)

def save_session_buffer(session_id: str):
    """Save session buffer to disk (encrypted)"""
    session = sessions.get(session_id)
    if not session:
        return
    
    buffer_file = session['buffer_file']
    try:
        with open(buffer_file, 'a') as f:
            for item in session['buffer']:
                # Encrypt each line
                line_id = str(uuid.uuid4())
                data = json.dumps(item, ensure_ascii=False)
                encrypted = simple_encrypt(data)
                entry = json.dumps({"id": line_id, "data": encrypted})
                f.write(entry + '\n')
        session['buffer'].clear()
    except Exception as e:
        print(f"Error saving buffer: {e}")

def load_session_buffer(session_id: str) -> List:
    """Load session buffer from disk"""
    session = sessions.get(session_id)
    if not session:
        return []
    
    buffer_file = session['buffer_file']
    if not buffer_file.exists():
        return []
    
    buffer = []
    try:
        with open(buffer_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                encrypted = entry.get('data', '')
                decrypted = simple_decrypt(encrypted)
                if decrypted:
                    buffer.append(json.loads(decrypted))
    except Exception as e:
        print(f"Error loading buffer: {e}")
    
    return buffer

def cleanup_session(session_id: str):
    """Clean up session (called on ack or disconnect)"""
    session = sessions.get(session_id)
    if not session:
        return
    
    # Remove buffer file
    try:
        buffer_file = session['buffer_file']
        if buffer_file.exists():
            buffer_file.unlink()
        # Remove session dir if empty
        session_dir = buffer_file.parent
        if session_dir.exists() and not any(session_dir.iterdir()):
            session_dir.rmdir()
    except Exception as e:
        print(f"Error cleaning session: {e}")
    
    sessions.pop(session_id, None)

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

def scan_models() -> List[Dict]:
    """Scan for available models in models/ and var/"""
    models = []
    
    # Scan models directory
    models_dir = Path(CONFIG['paths']['models_dir'])
    if models_dir.exists():
        for item in models_dir.iterdir():
            if item.is_dir():
                # Check if it looks like a model (has config.json)
                if (item / 'config.json').exists():
                    models.append({
                        "name": item.name,
                        "path": str(item),
                        "source": "models",
                        "size_mb": get_dir_size(item) / (1024 * 1024)
                    })
    
    # Scan var directory for trained models
    var_dir = Path(CONFIG['paths']['var_dir']) / 'models'
    if var_dir.exists():
        for item in var_dir.iterdir():
            if item.is_dir():
                if (item / 'config.json').exists():
                    models.append({
                        "name": item.name,
                        "path": str(item),
                        "source": "var",
                        "size_mb": get_dir_size(item) / (1024 * 1024)
                    })
    
    return models

def get_dir_size(path: Path) -> int:
    """Get directory size in bytes"""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except:
        pass
    return total

def auto_select_model() -> Optional[str]:
    """Auto-select model if only one available"""
    models = scan_models()
    if len(models) == 1:
        return models[0]['path']
    elif len(models) == 0:
        return None
    else:
        # Return first from models/ directory
        for m in models:
            if m['source'] == 'models':
                return m['path']
        return models[0]['path']

async def download_model_git_lfs(repo: str, target_dir: str, websocket=None):
    """Download model using git-lfs (background task)"""
    global available_models
    
    try:
        target_path = Path(CONFIG['paths']['models_dir']) / target_dir
        
        # Send log to client
        async def log(msg):
            if websocket:
                try:
                    await websocket.send(json.dumps({
                        "type": "log",
                        "message": msg
                    }))
                except:
                    pass
            print(msg)
        
        await log(f"Starting download: {repo}")
        
        # Check if git-lfs is available
        try:
            subprocess.run(['git', 'lfs', 'version'], check=True, capture_output=True)
        except:
            await log("Error: git-lfs not installed")
            return False
        
        # Clone with git-lfs
        await log("Cloning repository...")
        process = await asyncio.create_subprocess_exec(
            'git', 'clone', repo, str(target_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Stream output
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            await log(line.decode().strip())
        
        await process.wait()
        
        if process.returncode == 0:
            await log("Download complete!")
            # Rescan models
            available_models = scan_models()
            return True
        else:
            stderr = await process.stderr.read()
            await log(f"Error: {stderr.decode()}")
            return False
            
    except Exception as e:
        if websocket:
            try:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Download failed: {str(e)}"
                }))
            except:
                pass
        print(f"Download error: {e}")
        return False

def estimate_max_context(available_memory_gb: float, model_params_b: float) -> int:
    """Estimate maximum context length based on available memory"""
    # Rough estimation: 
    # Each token in context uses approximately 2 bytes per parameter (fp16)
    # Plus activation memory
    
    # Reserve some memory for model weights and activations
    usable_memory_gb = available_memory_gb * 0.6
    usable_memory_bytes = usable_memory_gb * 1024 * 1024 * 1024
    
    # Bytes per token in context (rough estimate)
    bytes_per_token = model_params_b * 1e9 * 2 / 1000  # Simplified
    
    max_tokens = int(usable_memory_bytes / bytes_per_token)
    
    # Cap at reasonable limits
    return min(max_tokens, CONFIG['limits']['max_context_length'])

def check_cuda_memory():
    """Check CUDA memory and estimate context limits"""
    if not torch.cuda.is_available():
        return None
    
    try:
        device_props = torch.cuda.get_device_properties(0)
        total_memory_gb = device_props.total_memory / (1024**3)
        
        # Get current memory usage
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        free = total_memory_gb - reserved
        
        return {
            "total_gb": total_memory_gb,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": free,
            "device": torch.cuda.get_device_name(0),
            "compute_capability": torch.cuda.get_device_capability(0)
        }
    except:
        return None

# ============================================================================
# MODEL LOADING
# ============================================================================

def check_cuda_compatibility():
    """Check CUDA version and architecture"""
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        major, minor = torch.cuda.get_device_capability()
        device_name = torch.cuda.get_device_name()
        
        print(f"CUDA Version: {cuda_version}")
        print(f"Device: {device_name}")
        print(f"Compute Capability: {major}.{minor}")
        
        if (major, minor) >= (10, 0):
            print("✓ Blackwell (SM 10.0+) architecture")
        elif (major, minor) >= (9, 0):
            print("✓ Hopper (SM 9.0) architecture")
        elif (major, minor) >= (8, 0):
            print("✓ Ampere/Ada (SM 8.0+) architecture")
        
        return major, minor
    return 0, 0

async def load_model(model_path: str):
    """Load model from path"""
    global model, tokenizer, processor, generation_config, compute_capability, use_flash_attention, current_model
    
    print(f"\nLoading model from: {model_path}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        # Check for safetensors
        safetensors_files = list(Path(model_path).glob("*.safetensors"))
        use_safetensors = len(safetensors_files) > 0
        
        # Check CUDA
        compute_capability = check_cuda_compatibility()
        use_flash_attention = (
            torch.cuda.is_available() and
            compute_capability[0] >= 8 and
            is_flash_attn_2_available() and
            CONFIG['optimization']['use_flash_attention']
        )
        
        # Determine dtype
        if CONFIG['optimization']['dtype'] == 'auto':
            torch_dtype = torch.bfloat16 if compute_capability[0] >= 8 else torch.float16
        else:
            torch_dtype = CONFIG['optimization']['dtype']
        
        attn_implementation = "flash_attention_2" if use_flash_attention else None
        
        print(f"  dtype: {torch_dtype}")
        print(f"  attention: {attn_implementation}")
        print(f"  safetensors: {use_safetensors}")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=CONFIG['model']['device_map'],
            local_files_only=True,
            trust_remote_code=CONFIG['model']['trust_remote_code'],
            attn_implementation=attn_implementation,
            use_safetensors=use_safetensors
        )
        
        # Try vision processor
        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            print("✓ Vision processor loaded")
        except:
            processor = None
        
        # Model info
        model_params = sum(p.numel() for p in model.parameters())
        model_params_b = model_params / 1e9
        
        print(f"✓ Model loaded: {Path(model_path).name}")
        print(f"  Parameters: {model_params_b:.2f}B")
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True
        )
        print("✓ Tokenizer loaded")
        
        # Generation config
        generation_config = GenerationConfig(
            **CONFIG['generation_defaults']
        )
        
        current_model = model_path
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

# ============================================================================
# INFERENCE
# ============================================================================

def parse_input_prompt(full_prompt):
    """Parse system and user prompts"""
    system_match = re.search(r"<\|system\|>(.*?)<\|end\|>", full_prompt, re.DOTALL)
    user_match = re.search(r"<\|user\|>(.*?)<\|end\|>", full_prompt, re.DOTALL)
    system_prompt = system_match.group(1).strip() if system_match else ""
    user_prompt = user_match.group(1).strip() if user_match else ""
    return system_prompt, user_prompt

def prompt2scheme(system_prompt, user_prompt):
    """Convert to chat template"""
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    except:
        sys_section = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n" if system_prompt else ""
        return f"[INST]{sys_section}{user_prompt}[/INST]"

async def stream_inference(websocket, session_id: str, prompt: str, config: Dict):
    """Stream inference tokens"""
    session = sessions[session_id]
    
    try:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs['input_ids'].shape[1]
        
        # Check context length
        memory_info = check_cuda_memory()
        if memory_info:
            max_context = estimate_max_context(
                memory_info['free_gb'],
                sum(p.numel() for p in model.parameters()) / 1e9
            )
            
            if input_length > max_context:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Context too long ({input_length} tokens). Max: {max_context}. Please retry with shorter input.",
                    "max_context": max_context,
                    "input_length": input_length
                }))
                return
        
        # Create generation config
        gen_config = GenerationConfig(**config)
        
        # Start message
        start_msg = {
            "type": "start",
            "tokens_in": input_length,
            "max_tokens": gen_config.max_new_tokens
        }
        await websocket.send(json.dumps(start_msg))
        session['buffer'].append(start_msg)
        
        # Generate
        generated_ids = []
        tokens_count = 0
        
        with torch.no_grad():
            past_key_values = None
            current_ids = inputs['input_ids']
            
            for _ in range(gen_config.max_new_tokens):
                outputs = model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                
                # Sampling
                if gen_config.do_sample:
                    logits = logits / gen_config.temperature
                    
                    # Top-k
                    if gen_config.top_k > 0:
                        top_k_values, _ = torch.topk(logits, gen_config.top_k)
                        logits[logits < top_k_values[:, -1, None]] = float('-inf')
                    
                    # Top-p
                    if gen_config.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > gen_config.top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = False
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits[:, indices_to_remove] = float('-inf')
                    
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                generated_ids.append(next_token.item())
                current_ids = next_token
                
                # Decode and stream
                token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                
                token_msg = {
                    "type": "token",
                    "data": token_text
                }
                await websocket.send(json.dumps(token_msg))
                session['buffer'].append(token_msg)
                
                tokens_count += 1
                
                # Check for EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                await asyncio.sleep(0)
        
        # Done
        done_msg = {
            "type": "done",
            "total_tokens": tokens_count
        }
        await websocket.send(json.dumps(done_msg))
        session['buffer'].append(done_msg)
        
        # Save buffer
        save_session_buffer(session_id)
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            error_msg = {
                "type": "error",
                "message": "CUDA out of memory. Context too long. Please retry with shorter input."
            }
        else:
            error_msg = {
                "type": "error",
                "message": str(e)
            }
        await websocket.send(json.dumps(error_msg))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ============================================================================
# MESSAGE HANDLERS
# ============================================================================

async def handle_message(websocket, message: Dict, session_id: str):
    """Handle incoming message"""
    global available_models
    
    session = sessions[session_id]
    msg_type = message.get('type', 'inference')
    
    # Update last activity
    session['last_activity'] = time.time()
    
    if msg_type == 'ping':
        await websocket.send(json.dumps({"type": "pong"}))
    
    elif msg_type == 'config':
        # Update session config
        config_data = message.get('data', {})
        session['config'].update(config_data)
        await websocket.send(json.dumps({
            "type": "config_ack",
            "config": session['config']
        }))
    
    elif msg_type == 'inference':
        # Parse prompt
        prompt_data = message.get('prompt', {})
        if isinstance(prompt_data, str):
            # Legacy format
            system_prompt, user_prompt = parse_input_prompt(prompt_data)
        else:
            system_prompt = prompt_data.get('system', '')
            user_prompt = prompt_data.get('user', '')
        
        full_prompt = prompt2scheme(system_prompt, user_prompt)
        
        # Use session config merged with request config
        config = session['config'].copy()
        config.update(message.get('config', {}))
        
        # Stream inference
        await stream_inference(websocket, session_id, full_prompt, config)
    
    elif msg_type == 'ack':
        # Client acknowledges receipt of response
        cleanup_session(session_id)
        await websocket.send(json.dumps({"type": "ack_received"}))
    
    elif msg_type == 'get_resources':
        # Return resource info
        memory_info = check_cuda_memory()
        resources = {
            "type": "resources",
            "cuda": memory_info,
            "model": {
                "name": Path(current_model).name if current_model else None,
                "path": current_model,
                "vision": processor is not None
            },
            "available_models": available_models
        }
        await websocket.send(json.dumps(resources))
    
    elif msg_type == 'list_models':
        # List available models
        available_models = scan_models()
        await websocket.send(json.dumps({
            "type": "models_list",
            "models": available_models
        }))
    
    elif msg_type == 'switch_model':
        # Switch to different model
        model_path = message.get('model_path')
        if model_path and Path(model_path).exists():
            success = await load_model(model_path)
            await websocket.send(json.dumps({
                "type": "model_switched" if success else "error",
                "message": f"Model switched to {model_path}" if success else "Failed to load model"
            }))
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Model not found"
            }))
    
    elif msg_type == 'download_model':
        # Download model via git-lfs (background)
        repo = message.get('repo')
        target_dir = message.get('target_dir')
        if repo and target_dir:
            asyncio.create_task(download_model_git_lfs(repo, target_dir, websocket))
            await websocket.send(json.dumps({
                "type": "download_started",
                "message": "Download started in background"
            }))
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Missing repo or target_dir"
            }))
    
    elif msg_type == 'delete_model':
        # Delete model from var/
        model_name = message.get('model_name')
        var_models_dir = Path(CONFIG['paths']['var_dir']) / 'models'
        model_path = var_models_dir / model_name
        
        if model_path.exists() and model_path.is_dir():
            try:
                shutil.rmtree(model_path)
                available_models = scan_models()
                await websocket.send(json.dumps({
                    "type": "model_deleted",
                    "message": f"Model {model_name} deleted"
                }))
            except Exception as e:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Failed to delete: {str(e)}"
                }))
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Model not found in var/"
            }))

# ============================================================================
# CONNECTION HANDLER
# ============================================================================

async def handle_connection(websocket, path):
    """Handle WebSocket connection"""
    client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    session_id = None
    
    try:
        # First message should contain session_id if reconnecting
        first_msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
        first_data = json.loads(first_msg)
        
        if 'session_id' in first_data and first_data['session_id'] in sessions:
            # Reconnect to existing session
            session_id = first_data['session_id']
            print(f"✓ Client reconnected: {client_addr} -> {session_id}")
            
            # Send buffered data if any
            buffered = load_session_buffer(session_id)
            if buffered:
                for item in buffered:
                    await websocket.send(json.dumps(item))
        else:
            # New session
            session_id = create_session()
            print(f"✓ Client connected: {client_addr} -> {session_id}")
        
        active_connections[websocket] = session_id
        
        # Send welcome
        memory_info = check_cuda_memory()
        max_context = None
        if memory_info and model:
            max_context = estimate_max_context(
                memory_info['free_gb'],
                sum(p.numel() for p in model.parameters()) / 1e9
            )
        
        welcome = {
            "type": "welcome",
            "session_id": session_id,
            "model": Path(current_model).name if current_model else None,
            "capabilities": {
                "vision": processor is not None,
                "flash_attention": use_flash_attention,
                "compute_capability": compute_capability,
                "max_context": max_context
            },
            "resources": memory_info
        }
        await websocket.send(json.dumps(welcome))
        
        # Message loop
        async for message in websocket:
            try:
                data = json.loads(message)
                await handle_message(websocket, data, session_id)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON"
                }))
            except Exception as e:
                print(f"Error handling message: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
    
    except asyncio.TimeoutError:
        print(f"✗ Client timeout: {client_addr}")
    
    except WebSocketException as e:
        print(f"✗ WebSocket error for {client_addr}: {e}")
    
    finally:
        # Cleanup (but keep session for potential reconnect)
        if websocket in active_connections:
            active_connections.pop(websocket)
        
        print(f"✗ Client disconnected: {client_addr}")
        
        try:
            await websocket.close()
        except:
            pass

# ============================================================================
# STARTUP
# ============================================================================

def cleanup_old_sessions():
    """Clean up sessions on server restart"""
    sessions_dir = Path(CONFIG['paths']['sessions_dir'])
    if sessions_dir.exists():
        for session_dir in sessions_dir.iterdir():
            if session_dir.is_dir():
                try:
                    shutil.rmtree(session_dir)
                except:
                    pass

async def main():
    """Main server"""
    print("=" * 60)
    print("LLMWS Enhanced Server")
    print("=" * 60)
    
    # Clean up old sessions
    cleanup_old_sessions()
    
    # Scan models
    global available_models
    available_models = scan_models()
    
    print(f"\nAvailable models: {len(available_models)}")
    for m in available_models:
        print(f"  - {m['name']} ({m['size_mb']:.0f} MB) [{m['source']}]")
    
    # Auto-select or use config
    model_path = CONFIG['model'].get('path')
    if not model_path:
        model_path = auto_select_model()
    
    if model_path and Path(model_path).exists():
        await load_model(model_path)
    else:
        print("\n⚠ No model loaded. Use client to download or switch model.")
    
    print("\n" + "=" * 60)
    print(f"Server ready on ws://{CONFIG['server']['host']}:{CONFIG['server']['port']}")
    print("=" * 60)
    
    async with websockets.serve(
        handle_connection,
        CONFIG['server']['host'],
        CONFIG['server']['port'],
        max_size=CONFIG['server']['max_message_size'],
        ping_interval=CONFIG['server']['ping_interval'],
        ping_timeout=CONFIG['server']['ping_timeout']
    ):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nServer shutdown by user")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")

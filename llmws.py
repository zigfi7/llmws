#!/usr/bin/env python3
"""
LLMWS Enhanced Server - WebSocket LLM Server
Features: Session persistence, model management, CUDA estimation, git-lfs support
KISS principle: Simple, clean, functional
"""
import os, sys, re, asyncio, json, base64, uuid, zlib, subprocess, shutil, time, warnings
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, Any, List, Set, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
import torch.nn.functional as F
import websockets
from websockets.exceptions import WebSocketException
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    AutoProcessor,
)
from transformers.utils import is_flash_attn_2_available
from safetensors.torch import save_file
from PIL import Image
from io import BytesIO

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None

# Suppress CUDA compatibility warnings for newer GPUs (like Blackwell GB10)
# These warnings are informational only - PyTorch works via backward compatibility
warnings.filterwarnings('ignore', message='.*CUDA capability.*is not compatible.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')

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
        "logs_dir": "var/logs",
        "datasets_dir": "var/datasets"
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

def save_config():
    """Save current config to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(CONFIG, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

CONFIG = load_config()

# Create directories
for dir_key in ['models_dir', 'var_dir', 'sessions_dir', 'logs_dir', 'datasets_dir']:
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

# Model status tracking
model_status = {
    "loaded": False,
    "status": "NOT_LOADED",  # NOT_LOADED, LOADING, OK, ERROR
    "name": None,
    "path": None,
    "error": None,
    "params": None,
    "max_context": None
}

# Session management
sessions: Dict[str, Dict] = {}  # session_id -> session data
active_connections: Dict[Any, str] = {}  # websocket -> session_id
request_queue: asyncio.Queue = asyncio.Queue()
processing_lock = asyncio.Lock()
current_model = None
train_lock = asyncio.Lock()
dataset_cache: Dict[str, str] = {}
train_status = {
    "running": False,
    "request_id": None,
    "started_at": None,
    "steps_done": 0,
    "max_steps": 0,
    "last_loss": None,
    "checkpoint": None,
    "error": None,
}

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
    """
    Estimate maximum context length based on available memory and model config.
    
    For multi-GPU setups, available_memory_gb is the sum of all GPUs.
    """
    
    # If model has loaded and we know its max context, use that as the ceiling
    if model_status.get("max_context"):
        model_max = model_status["max_context"]
    else:
        model_max = 128000  # Default to 128k if unknown
    
    # KV cache memory estimation:
    # For transformers, KV cache per token ≈ 2 * num_layers * hidden_size * 2 (key+value)
    # Rough approximation: ~1-2 KB per token for large models (70B)
    # For smaller models (8B): ~0.5 KB per token
    
    # Conservative estimate based on model size
    if model_params_b < 15:  # Small models (7B-13B)
        bytes_per_context_token = 512  # 0.5 KB
    elif model_params_b < 40:  # Medium models (13B-30B)
        bytes_per_context_token = 1024  # 1 KB
    else:  # Large models (70B+)
        bytes_per_context_token = 1536  # 1.5 KB
    
    # Reserve memory for activations and safety margin
    usable_memory_gb = available_memory_gb * 0.7  # Use 70% of free memory
    usable_memory_bytes = usable_memory_gb * 1024 * 1024 * 1024
    
    estimated_tokens = int(usable_memory_bytes / bytes_per_context_token)
    
    # Use the minimum of: estimated from memory, model's max, and config limit
    max_tokens = min(estimated_tokens, model_max, CONFIG['limits']['max_context_length'])
    
    # But if we have plenty of memory, prefer model's max
    if estimated_tokens > model_max * 1.5:
        return model_max  # Use model's full capability
    
    # Safety: never allow less than 1000 tokens
    return max(max_tokens, 1000)

def check_cuda_memory():
    """Check CUDA memory and estimate context limits (multi-GPU aware)"""
    if not torch.cuda.is_available():
        return None
    
    try:
        num_gpus = torch.cuda.device_count()
        
        # Multi-GPU: sum all available memory
        total_vram_gb = 0
        total_allocated_gb = 0
        total_reserved_gb = 0
        
        for device_id in range(num_gpus):
            props = torch.cuda.get_device_properties(device_id)
            total_vram_gb += props.total_memory / (1024**3)
            total_allocated_gb += torch.cuda.memory_allocated(device_id) / (1024**3)
            total_reserved_gb += torch.cuda.memory_reserved(device_id) / (1024**3)
        
        free_gb = total_vram_gb - total_reserved_gb
        
        return {
            "total_gb": total_vram_gb,
            "allocated_gb": total_allocated_gb,
            "reserved_gb": total_reserved_gb,
            "free_gb": free_gb,
            "num_gpus": num_gpus,
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

def _patch_default_rope_type_for_compat(model_config: Any) -> bool:
    """Patch rope_type=default for transformers versions that removed it."""
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        has_default = "default" in ROPE_INIT_FUNCTIONS
    except Exception:
        has_default = True

    if has_default:
        return False

    patched = False
    cfg_candidates = [model_config, getattr(model_config, "text_config", None)]
    for cfg in cfg_candidates:
        if cfg is None:
            continue
        rope_scaling = getattr(cfg, "rope_scaling", None)
        if isinstance(rope_scaling, dict) and rope_scaling.get("rope_type") == "default":
            new_rope = dict(rope_scaling)
            new_rope["rope_type"] = "linear"
            new_rope.setdefault("factor", 1.0)
            setattr(cfg, "rope_scaling", new_rope)
            patched = True

    if patched:
        print("⚠ Patched rope_type='default' -> 'linear' (factor=1.0) for compatibility")

    return patched

def _build_compatible_model_config(model_path: str):
    """Load and patch model config when runtime compatibility fixes are needed."""
    try:
        model_config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=CONFIG['model']['trust_remote_code'],
        )
    except Exception:
        return None

    _patch_default_rope_type_for_compat(model_config)
    return model_config

def _patch_processormixin_lenient_kwargs():
    """
    Some remote processors pass optional kwargs that newer ProcessorMixin rejects.
    Retry without unknown kwargs and assign them as attributes.
    """
    try:
        from transformers.processing_utils import ProcessorMixin
    except Exception:
        return

    if getattr(ProcessorMixin, "_llmws_lenient_kwargs_patch", False):
        return

    original_init = ProcessorMixin.__init__

    def _patched_init(self, *args, **kwargs):
        extra_kwargs = {}
        while True:
            try:
                result = original_init(self, *args, **kwargs)
                break
            except TypeError as err:
                msg = str(err)
                match = re.search(r"Unexpected keyword argument (.+?)\.", msg)
                if not match:
                    raise
                bad_key = match.group(1).strip().strip("'\"")
                if bad_key not in kwargs:
                    raise
                extra_kwargs[bad_key] = kwargs.pop(bad_key)

        for key, value in extra_kwargs.items():
            setattr(self, key, value)

        return result

    ProcessorMixin.__init__ = _patched_init
    ProcessorMixin._llmws_lenient_kwargs_patch = True

async def load_model(model_path: str):
    """Load model from path"""
    global model, tokenizer, processor, generation_config, compute_capability, use_flash_attention, current_model, model_status
    
    # Update status
    model_status["status"] = "LOADING"
    model_status["path"] = model_path
    model_status["name"] = Path(model_path).name
    model_status["error"] = None
    
    print(f"\nLoading model from: {model_path}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        # Check for safetensors
        safetensors_files = list(Path(model_path).glob("*.safetensors"))
        use_safetensors = len(safetensors_files) > 0
        
        # Check CUDA
        compute_capability = check_cuda_compatibility()
        
        # Blackwell GB10 (SM 12.x) - use SDPA with cuDNN backend (fastest on Blackwell)
        # PyTorch 2.10+cu130 has native cuDNN SDPA support for SM 12.x
        if compute_capability[0] >= 12:
            print("✓ Blackwell detected - using SDPA with cuDNN backend")
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            use_flash_attention = False
            attn_implementation = "sdpa"
        elif (torch.cuda.is_available() and
              compute_capability[0] >= 8 and
              is_flash_attn_2_available() and
              CONFIG['optimization']['use_flash_attention']):
            use_flash_attention = True
            attn_implementation = "flash_attention_2"
        elif compute_capability[0] >= 8:
            use_flash_attention = False
            attn_implementation = "sdpa"
        else:
            use_flash_attention = False
            attn_implementation = None

        # Determine dtype
        if CONFIG['optimization']['dtype'] == 'auto':
            torch_dtype = torch.bfloat16 if compute_capability[0] >= 8 else torch.float16
        else:
            torch_dtype = CONFIG['optimization']['dtype']

        print(f"  dtype: {torch_dtype}")
        print(f"  attention: {attn_implementation}")
        print(f"  safetensors: {use_safetensors}")

        model_kwargs = {
            "dtype": torch_dtype,
            "device_map": CONFIG['model']['device_map'],
            "local_files_only": True,
            "trust_remote_code": CONFIG['model']['trust_remote_code'],
            "attn_implementation": attn_implementation,
            "use_safetensors": use_safetensors,
        }
        compat_config = _build_compatible_model_config(model_path)
        if compat_config is not None:
            model_kwargs["config"] = compat_config

        # Load model with fallback for vision-language architectures like Molmo2.
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            print("✓ Loaded with AutoModelForCausalLM")
        except Exception as causal_err:
            if AutoModelForImageTextToText is None:
                raise causal_err
            print(f"⚠ AutoModelForCausalLM failed: {causal_err}")
            print("↪ Retrying with AutoModelForImageTextToText...")
            try:
                model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
            except TypeError:
                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs.pop("attn_implementation", None)
                fallback_kwargs.pop("use_safetensors", None)
                model = AutoModelForImageTextToText.from_pretrained(model_path, **fallback_kwargs)
            print("✓ Loaded with AutoModelForImageTextToText")
        
        # Try vision processor
        try:
            _patch_processormixin_lenient_kwargs()
            processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=False,
            )
            print("✓ Vision processor loaded")
        except Exception as proc_err:
            processor = None
            print(f"⚠ Vision processor unavailable: {proc_err}")
        
        # Model info
        model_params = sum(p.numel() for p in model.parameters())
        model_params_b = model_params / 1e9
        
        print(f"✓ Model loaded: {Path(model_path).name}")
        print(f"  Parameters: {model_params_b:.2f}B")
        
        # Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=CONFIG['model']['trust_remote_code'],
                use_fast=True
            )
            print("✓ Tokenizer loaded")
        except Exception as tok_err:
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                raise tok_err
            print(f"⚠ Tokenizer fallback from processor: {tok_err}")
        
        # Get model's max context from config
        try:
            model_config = model.config
            max_position_embeddings = getattr(model_config, 'max_position_embeddings', None)
            if max_position_embeddings:
                model_max_context = max_position_embeddings
                print(f"✓ Model max context: {model_max_context:,} tokens")
            else:
                model_max_context = 128000  # Default if unknown
                print(f"⚠ Model max context unknown, using: {model_max_context:,}")
        except:
            model_max_context = 128000
            print(f"⚠ Could not read model config, using default: {model_max_context:,}")
        
        # Generation config
        generation_config = GenerationConfig(
            **CONFIG['generation_defaults']
        )
        
        current_model = model_path
        
        # Check memory and estimate context
        memory_info = check_cuda_memory()
        if memory_info:
            print(f"\n💾 Memory Status:")
            print(f"  GPUs: {memory_info['num_gpus']}")
            print(f"  Total VRAM: {memory_info['total_gb']:.1f} GB")
            print(f"  Used by model: {memory_info['reserved_gb']:.1f} GB")
            print(f"  Free: {memory_info['free_gb']:.1f} GB")
            
            # Estimate max context
            estimated_context = estimate_max_context(memory_info['free_gb'], model_params_b)
            print(f"  Estimated max context: {estimated_context:,} tokens")
            
            if estimated_context < 1000:
                print(f"  ⚠️ WARNING: Low estimated context! Check memory usage.")
        
        # Update model status - SUCCESS
        model_status["loaded"] = True
        model_status["status"] = "OK"
        model_status["params"] = f"{model_params_b:.2f}B"
        model_status["max_context"] = model_max_context
        
        # Auto-save to config
        CONFIG['model']['path'] = model_path
        CONFIG['model']['name'] = Path(model_path).name
        CONFIG['model']['params'] = f"{model_params_b:.2f}B"
        CONFIG['model']['max_context'] = model_max_context
        CONFIG['model']['last_loaded'] = str(Path(model_path))
        save_config()
        
        print(f"✓ Model config saved")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        
        # Update model status - ERROR
        model_status["loaded"] = False
        model_status["status"] = "ERROR"
        model_status["error"] = str(e)
        
        # Save error to config
        CONFIG['model']['path'] = model_path
        CONFIG['model']['status'] = "ERROR"
        CONFIG['model']['error'] = str(e)
        save_config()
        
        return False

# ============================================================================
# TRAINING + SNAPSHOTS
# ============================================================================

def _sanitize_name(raw: Optional[str], default_prefix: str) -> str:
    seed = (raw or "").strip() or f"{default_prefix}_{int(time.time())}"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", seed).strip("-.")
    return cleaned or f"{default_prefix}_{int(time.time())}"

def _sanitize_request_id(request_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(request_id)).strip("-.")
    return cleaned or str(uuid.uuid4())

def _unique_subdir(base_dir: Path, name: str) -> Path:
    target = base_dir / name
    if not target.exists():
        return target
    idx = 1
    while True:
        candidate = base_dir / f"{name}_{idx}"
        if not candidate.exists():
            return candidate
        idx += 1

def _line_to_train_text(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped

    if not isinstance(obj, dict):
        return stripped

    input_keys = ("input", "prompt", "user", "instruction", "question", "text")
    target_keys = ("target", "response", "assistant", "output", "answer", "completion", "label")

    def pick(keys):
        for key in keys:
            value = obj.get(key)
            if value is None:
                continue
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, (int, float, bool)):
                return str(value)
        return ""

    source_text = pick(input_keys)
    target_text = pick(target_keys)

    if source_text and target_text:
        return f"{source_text}\n{target_text}"
    if target_text:
        return target_text
    if source_text:
        return source_text
    return stripped

class JsonlTrainDataset(Dataset):
    def __init__(self, dataset_path: Path):
        self.samples: List[str] = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                sample = _line_to_train_text(line)
                if sample:
                    self.samples.append(sample)
        if not self.samples:
            raise ValueError("Dataset is empty after parsing")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

def _resolve_train_tokenizer():
    train_tokenizer = tokenizer or getattr(processor, "tokenizer", None)
    if train_tokenizer is None:
        raise RuntimeError("Training requires tokenizer or processor.tokenizer")
    if train_tokenizer.pad_token_id is None and train_tokenizer.eos_token_id is not None:
        train_tokenizer.pad_token = train_tokenizer.eos_token
    return train_tokenizer

def _build_train_collate(train_tokenizer, max_seq_length: int):
    def collate(batch_texts: List[str]):
        encoded = train_tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return input_ids, attention_mask, labels
    return collate

def _snapshot_target_dir() -> Path:
    target = Path(CONFIG['paths']['var_dir']) / "models"
    target.mkdir(parents=True, exist_ok=True)
    return target

async def save_model_snapshot(name: Optional[str] = None) -> str:
    global available_models
    if model is None:
        raise RuntimeError("No model loaded")

    target_root = _snapshot_target_dir()
    snapshot_name = _sanitize_name(name, "snapshot")
    snapshot_path = _unique_subdir(target_root, snapshot_name)
    snapshot_path.mkdir(parents=True, exist_ok=False)

    try:
        # Save in safetensors format for portability and safety.
        state_dict = {
            key: value.detach().cpu().contiguous()
            for key, value in model.state_dict().items()
            if torch.is_tensor(value)
        }
        save_file(state_dict, str(snapshot_path / "model.safetensors"))

        if hasattr(model, "config") and model.config is not None:
            model.config.save_pretrained(str(snapshot_path))

        save_tokenizer = tokenizer or getattr(processor, "tokenizer", None)
        if save_tokenizer is not None:
            save_tokenizer.save_pretrained(str(snapshot_path))

        if processor is not None:
            try:
                processor.save_pretrained(str(snapshot_path))
            except Exception:
                pass

        available_models = scan_models()
        return str(snapshot_path)
    except Exception:
        try:
            shutil.rmtree(snapshot_path)
        except Exception:
            pass
        raise

async def _send_train_event(websocket, session_id: str, payload: Dict[str, Any]):
    await websocket.send(json.dumps(payload))
    session = sessions.get(session_id)
    if session is not None:
        session["buffer"].append(payload)

def _merge_train_config(custom: Dict[str, Any]) -> Dict[str, Any]:
    default_cfg = {
        "max_steps": 50,
        "batch_size": 1,
        "learning_rate": 5e-5,
        "max_seq_length": 2048,
        "log_every": 5,
        "save_every": 0,
        "gradient_accumulation": 1,
    }
    cfg = dict(default_cfg)
    cfg.update(custom or {})
    cfg["max_steps"] = max(1, int(cfg["max_steps"]))
    cfg["batch_size"] = max(1, int(cfg["batch_size"]))
    cfg["max_seq_length"] = max(32, int(cfg["max_seq_length"]))
    cfg["log_every"] = max(1, int(cfg["log_every"]))
    cfg["save_every"] = max(0, int(cfg["save_every"]))
    cfg["gradient_accumulation"] = max(1, int(cfg["gradient_accumulation"]))
    cfg["learning_rate"] = float(cfg["learning_rate"])
    return cfg

async def handle_train_request(websocket, session_id: str, message: Dict[str, Any]):
    request_id = str(message.get("request_id") or uuid.uuid4())
    dataset_text = message.get("dataset")
    resume = bool(message.get("resume", False))
    save_ckpt = bool(message.get("save_checkpoint", True))
    checkpoint_name = message.get("checkpoint_name")
    train_cfg = _merge_train_config(message.get("config", {}))

    if model is None:
        await _send_train_event(websocket, session_id, {
            "type": "error",
            "message": "No model loaded",
            "request_id": request_id,
        })
        return

    dataset_path: Optional[Path] = None
    rid_key = _sanitize_request_id(request_id)
    datasets_dir = Path(CONFIG["paths"]["datasets_dir"])
    datasets_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(dataset_text, str) and dataset_text.strip():
        dataset_path = datasets_dir / f"dataset_{rid_key}.jsonl"
        dataset_path.write_text(dataset_text, encoding="utf-8")
        dataset_cache[request_id] = str(dataset_path)
    elif resume:
        cached_path = dataset_cache.get(request_id)
        if cached_path and Path(cached_path).exists():
            dataset_path = Path(cached_path)
    if dataset_path is None:
        await _send_train_event(websocket, session_id, {
            "type": "error",
            "message": "No dataset provided and no cached dataset available for resume",
            "request_id": request_id,
        })
        return

    if train_lock.locked():
        await _send_train_event(websocket, session_id, {
            "type": "train_waiting",
            "request_id": request_id,
            "message": "Another training job is running, waiting for lock",
        })

    async with train_lock:
        train_status.update({
            "running": True,
            "request_id": request_id,
            "started_at": time.time(),
            "steps_done": 0,
            "max_steps": train_cfg["max_steps"],
            "last_loss": None,
            "checkpoint": None,
            "error": None,
        })

        await _send_train_event(websocket, session_id, {
            "type": "train_started",
            "request_id": request_id,
            "config": train_cfg,
            "dataset_path": str(dataset_path),
        })

        model_config = getattr(model, "config", None)
        had_use_cache = hasattr(model_config, "use_cache") if model_config is not None else False
        previous_use_cache = getattr(model_config, "use_cache", None) if model_config is not None else None
        if model_config is not None:
            try:
                setattr(model_config, "use_cache", False)
            except Exception:
                pass

        try:
            train_tokenizer = _resolve_train_tokenizer()
            dataset = JsonlTrainDataset(dataset_path)
            collate = _build_train_collate(train_tokenizer, train_cfg["max_seq_length"])
            loader = DataLoader(
                dataset,
                batch_size=train_cfg["batch_size"],
                shuffle=True,
                collate_fn=collate,
            )

            params = [p for p in model.parameters() if p.requires_grad]
            if not params:
                raise RuntimeError("Model exposes no trainable parameters")
            optimizer = torch.optim.AdamW(params, lr=train_cfg["learning_rate"])

            steps_done = 0
            model.train()

            while steps_done < train_cfg["max_steps"]:
                for batch in loader:
                    steps_done += 1
                    input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]
                    with torch.enable_grad():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )

                        raw_loss = getattr(outputs, "loss", None)
                        logits = getattr(outputs, "logits", None)

                        # Some multimodal backbones can return detached loss tensors.
                        # Fallback to explicit CE loss over shifted logits.
                        if raw_loss is None or not raw_loss.requires_grad:
                            if logits is None:
                                raise RuntimeError("Model returned no logits for training")
                            if logits.size(1) < 2:
                                raise RuntimeError("Not enough tokens to compute training loss")
                            shift_logits = logits[:, :-1, :].contiguous()
                            shift_labels = labels[:, 1:].contiguous()
                            raw_loss = F.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                                ignore_index=-100,
                            )

                        if not raw_loss.requires_grad:
                            logits_grad = getattr(logits, "requires_grad", None)
                            raise RuntimeError(
                                "Training loss has no grad_fn "
                                f"(loss.requires_grad={raw_loss.requires_grad}, logits.requires_grad={logits_grad})"
                            )

                        loss = raw_loss / train_cfg["gradient_accumulation"]
                    loss.backward()

                    if steps_done % train_cfg["gradient_accumulation"] == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    if steps_done % train_cfg["log_every"] == 0:
                        loss_value = float(raw_loss.item())
                        train_status["steps_done"] = steps_done
                        train_status["last_loss"] = loss_value
                        await _send_train_event(websocket, session_id, {
                            "type": "train_progress",
                            "request_id": request_id,
                            "step": steps_done,
                            "max_steps": train_cfg["max_steps"],
                            "loss": loss_value,
                        })

                    if train_cfg["save_every"] and save_ckpt and (steps_done % train_cfg["save_every"] == 0):
                        periodic_name = f"{_sanitize_name(checkpoint_name, 'train')}_step_{steps_done}"
                        periodic_checkpoint = await save_model_snapshot(periodic_name)
                        await _send_train_event(websocket, session_id, {
                            "type": "train_checkpoint",
                            "request_id": request_id,
                            "step": steps_done,
                            "path": periodic_checkpoint,
                        })

                    if steps_done >= train_cfg["max_steps"]:
                        break
                    await asyncio.sleep(0)

            if steps_done % train_cfg["gradient_accumulation"] != 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            model.eval()
            checkpoint_path = None
            if save_ckpt:
                checkpoint_path = await save_model_snapshot(
                    _sanitize_name(checkpoint_name, f"train_{rid_key}")
                )

            train_status.update({
                "running": False,
                "steps_done": steps_done,
                "last_loss": train_status.get("last_loss"),
                "checkpoint": checkpoint_path,
                "error": None,
            })

            await _send_train_event(websocket, session_id, {
                "type": "train_done",
                "request_id": request_id,
                "steps": steps_done,
                "checkpoint": checkpoint_path,
            })
            save_session_buffer(session_id)

        except Exception as err:
            train_status.update({
                "running": False,
                "steps_done": train_status.get("steps_done", 0),
                "error": str(err),
            })
            await _send_train_event(websocket, session_id, {
                "type": "error",
                "request_id": request_id,
                "message": f"Training failed: {err}",
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            if model_config is not None:
                try:
                    if had_use_cache:
                        setattr(model_config, "use_cache", previous_use_cache)
                    elif hasattr(model_config, "use_cache"):
                        delattr(model_config, "use_cache")
                except Exception:
                    pass
            train_status["running"] = False

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

def _decode_image_data_url(data_url: str) -> bytes:
    if "," not in data_url:
        raise ValueError("data_url is missing comma separator")
    _, payload = data_url.split(",", 1)
    payload = "".join(payload.split())
    return base64.b64decode(payload)

def _load_image_from_bytes(raw: bytes, label: str) -> Image.Image:
    try:
        with Image.open(BytesIO(raw)) as img:
            return img.convert("RGB")
    except Exception as err:
        raise ValueError(f"Failed to decode image '{label}': {err}") from err

def _load_image_from_entry(entry: Any, index: int) -> Image.Image:
    label = f"image[{index}]"

    # Allow simple string entries:
    # - data URL
    # - filesystem path
    # - raw base64 payload
    if isinstance(entry, str):
        text = entry.strip()
        if text.startswith("data:image/"):
            return _load_image_from_bytes(_decode_image_data_url(text), label)
        if Path(text).exists():
            with Image.open(text) as img:
                return img.convert("RGB")
        return _load_image_from_bytes(base64.b64decode("".join(text.split())), label)

    if not isinstance(entry, dict):
        raise ValueError(f"{label} has unsupported type: {type(entry).__name__}")

    source = entry.get("source")
    payload = source if isinstance(source, dict) else entry
    if not isinstance(payload, dict):
        raise ValueError(f"{label} source is not an object")

    data_url = payload.get("data_url")
    if isinstance(data_url, str) and data_url.startswith("data:image/"):
        return _load_image_from_bytes(_decode_image_data_url(data_url), label)

    b64 = payload.get("base64")
    if isinstance(b64, str) and b64.strip():
        return _load_image_from_bytes(base64.b64decode("".join(b64.split())), label)

    data = payload.get("data")
    if isinstance(data, str) and data.strip():
        if data.startswith("data:image/"):
            return _load_image_from_bytes(_decode_image_data_url(data), label)
        return _load_image_from_bytes(base64.b64decode("".join(data.split())), label)

    path_value = payload.get("path")
    if isinstance(path_value, str) and path_value.strip():
        path_obj = Path(path_value)
        if not path_obj.exists():
            raise ValueError(f"{label} path does not exist: {path_value}")
        with Image.open(path_obj) as img:
            return img.convert("RGB")

    url_value = payload.get("url")
    if isinstance(url_value, str) and url_value.strip():
        parsed = urlparse(url_value)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"{label} url scheme not supported: {parsed.scheme}")
        with urlopen(url_value, timeout=10) as resp:
            raw = resp.read()
        return _load_image_from_bytes(raw, label)

    raise ValueError(f"{label} is missing image source (data_url/base64/path/url)")

def _extract_inline_data_url_images(user_prompt: str) -> Tuple[str, List[Image.Image]]:
    if not user_prompt:
        return user_prompt, []

    block_re = re.compile(r"\[IMAGE FILE:[^\]]*\](.*?)\[/IMAGE FILE\]", re.IGNORECASE | re.DOTALL)
    data_url_re = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+")
    images: List[Image.Image] = []

    for block_match in block_re.finditer(user_prompt):
        inner = block_match.group(1) or ""
        data_urls = data_url_re.findall(inner)
        for raw_data_url in data_urls:
            try:
                images.append(_load_image_from_bytes(_decode_image_data_url(raw_data_url), "inline"))
            except Exception as err:
                print(f"⚠ Skipping inline image block: {err}")

    cleaned_prompt = block_re.sub("", user_prompt).strip()
    return cleaned_prompt, images

def parse_prompt_images(prompt_data: Any, message_images: Any, user_prompt: str) -> Tuple[str, List[Image.Image]]:
    explicit_entries: List[Any] = []

    if isinstance(prompt_data, dict):
        prompt_images = prompt_data.get("images")
        if isinstance(prompt_images, list):
            explicit_entries.extend(prompt_images)

    if isinstance(message_images, list):
        explicit_entries.extend(message_images)

    images: List[Image.Image] = []
    for idx, entry in enumerate(explicit_entries):
        images.append(_load_image_from_entry(entry, idx))

    cleaned_prompt, inline_images = _extract_inline_data_url_images(user_prompt)
    images.extend(inline_images)
    return cleaned_prompt, images

def _build_multimodal_messages(system_prompt: str, user_prompt: str, image_count: int) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []

    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

    user_content: List[Dict[str, str]] = []
    if user_prompt:
        user_content.append({"type": "text", "text": user_prompt})
    if not user_content:
        user_content.append({"type": "text", "text": "Describe the provided image(s)."})
    for _ in range(image_count):
        user_content.append({"type": "image"})

    messages.append({"role": "user", "content": user_content})
    return messages

def _prepare_model_inputs(
    system_prompt: str,
    user_prompt: str,
    images: List[Image.Image],
) -> Tuple[Any, int, bool]:
    if images:
        if processor is None:
            raise RuntimeError("This model/session does not expose a vision processor")

        mm_messages = _build_multimodal_messages(system_prompt, user_prompt, len(images))
        rendered_prompt = processor.apply_chat_template(
            mm_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        try:
            model_inputs = processor(text=rendered_prompt, images=images, return_tensors="pt")
        except TypeError:
            model_inputs = processor(rendered_prompt, images=images, return_tensors="pt")

        if hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(device)
        else:
            model_inputs = {
                key: (value.to(device) if hasattr(value, "to") else value)
                for key, value in dict(model_inputs).items()
            }

        input_ids = model_inputs.get("input_ids")
        input_length = int(input_ids.shape[1]) if input_ids is not None else 0
        return model_inputs, input_length, True

    full_prompt = prompt2scheme(system_prompt, user_prompt)
    if tokenizer is not None:
        model_inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        input_length = int(model_inputs["input_ids"].shape[1])
        return model_inputs, input_length, False

    if processor is None:
        raise RuntimeError("No tokenizer or processor available for text inference")

    try:
        model_inputs = processor(text=full_prompt, return_tensors="pt")
    except TypeError:
        model_inputs = processor(full_prompt, return_tensors="pt")

    if hasattr(model_inputs, "to"):
        model_inputs = model_inputs.to(device)
    else:
        model_inputs = {
            key: (value.to(device) if hasattr(value, "to") else value)
            for key, value in dict(model_inputs).items()
        }

    input_ids = model_inputs.get("input_ids")
    input_length = int(input_ids.shape[1]) if input_ids is not None else 0
    return model_inputs, input_length, False

async def stream_inference(
    websocket,
    session_id: str,
    system_prompt: str,
    user_prompt: str,
    config: Dict,
    images: Optional[List[Image.Image]] = None,
):
    """Stream inference tokens"""
    session = sessions[session_id]
    images = images or []
    
    try:
        model_inputs, input_length, multimodal = _prepare_model_inputs(
            system_prompt,
            user_prompt,
            images,
        )
        
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
        if multimodal and getattr(gen_config, "do_sample", False):
            # Molmo2 + current CUDA/transformers stack can hit device-side asserts in sampling.
            # Use greedy decode for multimodal requests to keep server stable.
            gen_config.do_sample = False
            if hasattr(gen_config, "temperature"):
                gen_config.temperature = 1.0
            if hasattr(gen_config, "top_p"):
                gen_config.top_p = 1.0
            if hasattr(gen_config, "top_k"):
                gen_config.top_k = 50
        if multimodal and hasattr(gen_config, "repetition_penalty"):
            # Repetition penalty can trigger scatter-gather asserts in this runtime combo.
            gen_config.repetition_penalty = 1.0
        
        # Start message
        start_msg = {
            "type": "start",
            "tokens_in": input_length,
            "max_tokens": gen_config.max_new_tokens,
            "images_in": len(images),
            "multimodal": multimodal,
            "do_sample": bool(getattr(gen_config, "do_sample", False)),
        }
        await websocket.send(json.dumps(start_msg))
        session['buffer'].append(start_msg)

        # Multimodal path: use model.generate and send output as chunk(s).
        if multimodal:
            with torch.no_grad():
                generated = model.generate(
                    **model_inputs,
                    generation_config=gen_config,
                )

            if hasattr(generated, "sequences"):
                generated = generated.sequences

            prompt_tokens = model_inputs.get("input_ids")
            prompt_len = int(prompt_tokens.shape[1]) if prompt_tokens is not None else 0
            output_ids = generated[0][prompt_len:]
            decode_tokenizer = tokenizer or getattr(processor, "tokenizer", None)
            if decode_tokenizer is None:
                raise RuntimeError("No tokenizer available for decoding multimodal output")
            decoded = decode_tokenizer.decode(output_ids, skip_special_tokens=True)

            if decoded:
                token_msg = {"type": "token", "data": decoded}
                await websocket.send(json.dumps(token_msg))
                session['buffer'].append(token_msg)

            tokens_count = int(output_ids.shape[0]) if hasattr(output_ids, "shape") else 0
        else:
            # Text-only fast streaming with incremental decode window.
            multilingual_separators = {
                ' ', '\t', '\n',
                ',', '.', '!', '?', ';', ':',
                '，', '。', '！', '？', '；', '：', '、', '…', '—', '～', '-',
                '¡', '¿', '。', '｡', '·', '‧', '؟', '؛'
            }
            decode_window = 20
            id_buffer = []
            sent_output = ""
            last_decode_idx = 0
            tokens_count = 0

            with torch.no_grad():
                past_key_values = None
                current_ids = model_inputs["input_ids"]

                for _ in range(gen_config.max_new_tokens):
                    outputs = model(
                        input_ids=current_ids,
                        past_key_values=past_key_values,
                        use_cache=True
                    )

                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values

                    if gen_config.do_sample:
                        logits = logits / gen_config.temperature

                        if gen_config.top_k > 0:
                            top_k_values, _ = torch.topk(logits, gen_config.top_k)
                            logits[logits < top_k_values[:, -1, None]] = float('-inf')

                        if gen_config.top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(
                                torch.softmax(sorted_logits, dim=-1),
                                dim=-1,
                            )
                            sorted_indices_to_remove = cumulative_probs > gen_config.top_p
                            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                            sorted_indices_to_remove[:, 0] = False
                            indices_to_remove = sorted_indices[sorted_indices_to_remove]
                            logits[:, indices_to_remove] = float('-inf')

                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)

                    if next_token.item() == tokenizer.eos_token_id:
                        if len(id_buffer) > last_decode_idx:
                            remaining_tokens = id_buffer[last_decode_idx:]
                            remaining_text = tokenizer.decode(remaining_tokens, skip_special_tokens=True)
                            new_text = remaining_text
                            if sent_output:
                                for overlap_len in range(min(50, len(sent_output)), 0, -1):
                                    if remaining_text.startswith(sent_output[-overlap_len:]):
                                        new_text = remaining_text[overlap_len:]
                                        break
                            if new_text:
                                token_msg = {"type": "token", "data": new_text}
                                await websocket.send(json.dumps(token_msg))
                                session['buffer'].append(token_msg)
                        break

                    id_buffer.append(next_token.item())
                    current_ids = next_token
                    tokens_count += 1

                    window_start = max(0, len(id_buffer) - decode_window)
                    window_tokens = id_buffer[window_start:]
                    window_text = tokenizer.decode(window_tokens, skip_special_tokens=True)
                    new_text = window_text

                    if sent_output and window_start < last_decode_idx:
                        overlap_search_len = min(100, len(sent_output))
                        sent_tail = sent_output[-overlap_search_len:]
                        best_match_pos = -1
                        for check_len in range(min(len(window_text), len(sent_tail)), 0, -1):
                            if sent_tail.endswith(window_text[:check_len]):
                                best_match_pos = check_len
                                break
                        if best_match_pos > 0:
                            new_text = window_text[best_match_pos:]

                    last_separator_pos = -1
                    for idx in reversed(range(len(new_text))):
                        if new_text[idx] in multilingual_separators:
                            last_separator_pos = idx + 1
                            break

                    if last_separator_pos > 0:
                        to_send = new_text[:last_separator_pos]
                        if to_send:
                            token_msg = {"type": "token", "data": to_send}
                            await websocket.send(json.dumps(token_msg))
                            session['buffer'].append(token_msg)
                            sent_output += to_send
                            last_decode_idx = len(id_buffer)

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
        if train_lock.locked():
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Training in progress. Inference is temporarily unavailable."
            }))
            return

        # Parse prompt
        prompt_data = message.get('prompt', {})
        top_level_images = message.get('images')
        if isinstance(prompt_data, str):
            # Legacy format
            system_prompt, user_prompt = parse_input_prompt(prompt_data)
        else:
            system_prompt = prompt_data.get('system', '')
            user_prompt = prompt_data.get('user', '')

        user_prompt, images = parse_prompt_images(prompt_data, top_level_images, user_prompt)
        
        # Use session config merged with request config
        config = session['config'].copy()
        config.update(message.get('config', {}))
        
        # Stream inference
        await stream_inference(
            websocket,
            session_id,
            system_prompt,
            user_prompt,
            config,
            images=images,
        )
    
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
            "training": train_status,
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
                "message": f"Model switched to {model_path}" if success else "Failed to load model",
                "status": model_status
            }))
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Model not found"
            }))
    
    elif msg_type == 'recheck_model':
        # Recheck current model (useful if files were restored)
        if current_model and Path(current_model).exists():
            print(f"\n[Recheck] Reloading model: {current_model}")
            success = await load_model(current_model)
            await websocket.send(json.dumps({
                "type": "recheck_complete",
                "success": success,
                "status": model_status
            }))
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "No model loaded or model files not found. Please check files and try switching model."
            }))
    
    elif msg_type == 'get_status':
        # Get current model status
        memory_info = check_cuda_memory()
        await websocket.send(json.dumps({
            "type": "status",
            "model": model_status,
            "memory": memory_info,
            "sessions": len(sessions),
            "training": train_status
        }))

    elif msg_type == 'train_status':
        await websocket.send(json.dumps({
            "type": "train_status",
            "training": train_status
        }))

    elif msg_type == 'save_snapshot':
        snapshot_name = message.get("name")
        try:
            path = await save_model_snapshot(snapshot_name)
            await websocket.send(json.dumps({
                "type": "snapshot_saved",
                "path": path
            }))
        except Exception as err:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Snapshot failed: {err}"
            }))

    elif msg_type == 'train':
        await handle_train_request(websocket, session_id, message)

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

async def handle_connection(websocket):
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
                "sdpa": not use_flash_attention and compute_capability[0] >= 8,
                "compute_capability": compute_capability,
                "max_context": max_context,
                "training": True,
            },
            "resources": memory_info,
            "training": train_status
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

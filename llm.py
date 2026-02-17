#!/usr/bin/env python3
"""
LLMWS Enhanced Client
Supports: session persistence, model management, resource queries
"""
import asyncio
import websockets
import json
import sys
import argparse
import base64
from pathlib import Path
from typing import Optional, List, Dict

class LLMWSEnhancedClient:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}"
        self.websocket = None
        self.session_id = None
        self.config = {
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True
        }
    
    async def connect(self, session_id: Optional[str] = None):
        """Connect to server (optionally resume session)"""
        try:
            self.websocket = await websockets.connect(
                self.uri,
                max_size=2**24,
                ping_interval=30,
                ping_timeout=10
            )
            
            # Send session_id if reconnecting
            hello = {"session_id": session_id} if session_id else {}
            await self.websocket.send(json.dumps(hello))
            
            # Receive welcome
            welcome = await self.websocket.recv()
            welcome_data = json.loads(welcome)
            
            if welcome_data.get('type') == 'welcome':
                self.session_id = welcome_data.get('session_id')
                print(f"\n{'='*60}")
                print(f"Connected to: {welcome_data.get('model', 'No model loaded')}")
                print(f"Session ID: {self.session_id}")
                
                caps = welcome_data.get('capabilities', {})
                print(f"Capabilities:")
                print(f"  Vision: {caps.get('vision', False)}")
                print(f"  Flash Attention: {caps.get('flash_attention', False)}")
                
                max_ctx = caps.get('max_context')
                if max_ctx:
                    print(f"  Max Context: {max_ctx} tokens")
                
                resources = welcome_data.get('resources')
                if resources:
                    print(f"  GPU: {resources.get('device', 'N/A')}")
                    print(f"  Free VRAM: {resources.get('free_gb', 0):.1f} GB")
                
                print(f"{'='*60}\n")
                return True
            
            return False
            
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    async def disconnect(self, send_ack: bool = True):
        """Disconnect from server"""
        if self.websocket:
            if send_ack:
                try:
                    await self.websocket.send(json.dumps({"type": "ack"}))
                    # Wait for ack_received
                    response = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=2.0
                    )
                except:
                    pass
            
            await self.websocket.close()
            print("\n✓ Disconnected")
    
    async def set_config(self, **kwargs):
        """Update inference configuration"""
        self.config.update(kwargs)
        
        message = {
            "type": "config",
            "data": self.config
        }
        
        await self.websocket.send(json.dumps(message))
        
        response = await self.websocket.recv()
        response_data = json.loads(response)
        
        if response_data.get('type') == 'config_ack':
            print(f"✓ Config updated: {response_data.get('config')}")
            return True
        
        return False
    
    async def inference(self, prompt: str, system: str = "", images: Optional[List[Dict[str, str]]] = None):
        """Run inference with streaming"""
        prompt_payload = {
            "system": system,
            "user": prompt
        }
        if images:
            prompt_payload["images"] = images

        message = {
            "type": "inference",
            "prompt": prompt_payload,
            "config": self.config
        }
        
        await self.websocket.send(json.dumps(message))
        
        print("\n" + "─" * 60)
        print("Assistant:", end=" ", flush=True)
        
        full_response = ""
        tokens_count = 0
        
        try:
            async for response in self.websocket:
                data = json.loads(response)
                msg_type = data.get('type')
                
                if msg_type == 'start':
                    continue
                
                elif msg_type == 'token':
                    token_text = data.get('data', '')
                    print(token_text, end='', flush=True)
                    full_response += token_text
                    tokens_count += 1
                
                elif msg_type == 'done':
                    total_tokens = data.get('total_tokens', 0)
                    print(f"\n{'─' * 60}")
                    print(f"Tokens: {total_tokens}")
                    break
                
                elif msg_type == 'error':
                    print(f"\n✗ Error: {data.get('message')}")
                    break
        
        except Exception as e:
            print(f"\n✗ Streaming error: {e}")
        
        return full_response
    
    async def get_resources(self):
        """Get server resources"""
        await self.websocket.send(json.dumps({"type": "get_resources"}))
        
        response = await self.websocket.recv()
        data = json.loads(response)
        
        if data.get('type') == 'resources':
            print("\n" + "="*60)
            print("Server Resources")
            print("="*60)
            
            cuda = data.get('cuda')
            if cuda:
                print(f"\nGPU: {cuda.get('device')}")
                print(f"  Total: {cuda.get('total_gb'):.1f} GB")
                print(f"  Free: {cuda.get('free_gb'):.1f} GB")
                print(f"  Compute: SM {cuda.get('compute_capability', (0,0))[0]}.{cuda.get('compute_capability', (0,0))[1]}")
            
            model = data.get('model', {})
            print(f"\nCurrent Model: {model.get('name', 'None')}")
            print(f"  Vision: {model.get('vision', False)}")
            
            models = data.get('available_models', [])
            if models:
                print(f"\nAvailable Models: {len(models)}")
                for m in models:
                    print(f"  - {m['name']} ({m['size_mb']:.0f} MB) [{m['source']}]")
            
            print("="*60 + "\n")
    
    async def list_models(self):
        """List available models"""
        await self.websocket.send(json.dumps({"type": "list_models"}))
        
        response = await self.websocket.recv()
        data = json.loads(response)
        
        if data.get('type') == 'models_list':
            models = data.get('models', [])
            print("\n" + "="*60)
            print(f"Available Models: {len(models)}")
            print("="*60)
            
            for i, m in enumerate(models, 1):
                print(f"\n{i}. {m['name']}")
                print(f"   Path: {m['path']}")
                print(f"   Size: {m['size_mb']:.0f} MB")
                print(f"   Source: {m['source']}")
            
            print("="*60 + "\n")
            return models
        
        return []
    
    async def switch_model(self, model_path: str):
        """Switch to a different model"""
        await self.websocket.send(json.dumps({
            "type": "switch_model",
            "model_path": model_path
        }))
        
        # Listen for logs
        async for response in self.websocket:
            data = json.loads(response)
            msg_type = data.get('type')
            
            if msg_type == 'log':
                print(data.get('message'))
            elif msg_type == 'model_switched':
                print(f"\n✓ {data.get('message')}")
                break
            elif msg_type == 'error':
                print(f"\n✗ {data.get('message')}")
                break
    
    async def download_model(self, repo: str, target_dir: str):
        """Download model via git-lfs"""
        await self.websocket.send(json.dumps({
            "type": "download_model",
            "repo": repo,
            "target_dir": target_dir
        }))
        
        print("\nDownloading model (this may take a while)...\n")
        
        # Listen for logs
        async for response in self.websocket:
            data = json.loads(response)
            msg_type = data.get('type')
            
            if msg_type == 'log':
                print(data.get('message'))
            elif msg_type == 'download_started':
                print("Download started in background")
                print("You can continue using the client")
                break
            elif msg_type == 'error':
                print(f"✗ {data.get('message')}")
                break
    
    async def delete_model(self, model_name: str):
        """Delete model from var/"""
        await self.websocket.send(json.dumps({
            "type": "delete_model",
            "model_name": model_name
        }))
        
        response = await self.websocket.recv()
        data = json.loads(response)
        
        if data.get('type') == 'model_deleted':
            print(f"✓ {data.get('message')}")
        else:
            print(f"✗ {data.get('message')}")

    async def save_snapshot(self, name: Optional[str] = None):
        """Request server to save model snapshot"""
        payload = {"type": "save_snapshot"}
        if name:
            payload["name"] = name
        await self.websocket.send(json.dumps(payload))

        response = await self.websocket.recv()
        data = json.loads(response)
        if data.get("type") == "snapshot_saved":
            print(f"✓ Snapshot saved: {data.get('path')}")
            return data.get("path")

        print(f"✗ {data.get('message')}")
        return None

    async def get_train_status(self):
        """Fetch current training status"""
        await self.websocket.send(json.dumps({"type": "train_status"}))
        response = await self.websocket.recv()
        data = json.loads(response)
        if data.get("type") != "train_status":
            print(f"✗ {data.get('message', 'Unexpected response')}")
            return None

        status = data.get("training", {})
        print("\n" + "=" * 60)
        print("Training Status")
        print("=" * 60)
        print(f"Running: {status.get('running')}")
        print(f"Request ID: {status.get('request_id')}")
        print(f"Steps: {status.get('steps_done')}/{status.get('max_steps')}")
        print(f"Last loss: {status.get('last_loss')}")
        print(f"Checkpoint: {status.get('checkpoint')}")
        print(f"Error: {status.get('error')}")
        print("=" * 60 + "\n")
        return status
    
    async def interactive_mode(self):
        """Interactive chat mode"""
        print("\n" + "="*60)
        print("Interactive Mode")
        print("Commands:")
        print("  /config <key>=<value>  - Update config")
        print("  /system <text>         - Set system prompt")
        print("  /resources             - Show server resources")
        print("  /models                - List available models")
        print("  /switch <path>         - Switch model")
        print("  /download <repo> <dir> - Download model")
        print("  /delete <name>         - Delete model from var/")
        print("  /snapshot [name]       - Save model snapshot")
        print("  /train_status          - Show training status")
        print("  /session               - Show session ID")
        print("  /quit or /exit         - Exit")
        print("="*60 + "\n")
        
        system_prompt = ""
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    parts = user_input.split(maxsplit=1)
                    command = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else ""
                    
                    if command in ['/quit', '/exit']:
                        break
                    
                    elif command == '/config':
                        if '=' in args:
                            key, value = args.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            try:
                                if value.lower() in ['true', 'false']:
                                    value = value.lower() == 'true'
                                elif '.' in value:
                                    value = float(value)
                                else:
                                    value = int(value)
                            except ValueError:
                                pass
                            
                            await self.set_config(**{key: value})
                        else:
                            print(f"Current config: {self.config}")
                    
                    elif command == '/system':
                        system_prompt = args
                        print(f"✓ System prompt set")
                    
                    elif command == '/resources':
                        await self.get_resources()
                    
                    elif command == '/models':
                        await self.list_models()
                    
                    elif command == '/switch':
                        if args:
                            await self.switch_model(args)
                        else:
                            print("Usage: /switch <model_path>")
                    
                    elif command == '/download':
                        parts = args.split(maxsplit=1)
                        if len(parts) == 2:
                            await self.download_model(parts[0], parts[1])
                        else:
                            print("Usage: /download <repo_url> <target_dir>")
                    
                    elif command == '/delete':
                        if args:
                            confirm = input(f"Delete model '{args}'? (y/n): ")
                            if confirm.lower() == 'y':
                                await self.delete_model(args)
                        else:
                            print("Usage: /delete <model_name>")

                    elif command == '/snapshot':
                        await self.save_snapshot(args if args else None)

                    elif command in ['/train_status', '/trainstatus']:
                        await self.get_train_status()
                    
                    elif command == '/session':
                        print(f"Session ID: {self.session_id}")
                    
                    else:
                        print(f"✗ Unknown command: {command}")
                    
                    continue
                
                # Regular inference
                await self.inference(
                    prompt=user_input,
                    system=system_prompt
                )
                
            except KeyboardInterrupt:
                print("\n\nUse /quit to exit")
                continue
            except EOFError:
                break
            except Exception as e:
                print(f"\n✗ Error: {e}")
                continue
        
        print("\nGoodbye!")

async def main():
    parser = argparse.ArgumentParser(description="LLMWS Enhanced Client")
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8765, help='Server port')
    parser.add_argument('-p', '--prompt', help='User prompt')
    parser.add_argument('-s', '--system', default='', help='System prompt')
    parser.add_argument('--image', action='append', default=[], help='Image path (repeatable)')
    parser.add_argument('--session', help='Session ID to reconnect')
    parser.add_argument('--temperature', type=float, help='Temperature')
    parser.add_argument('--top-p', type=float, help='Top-p')
    parser.add_argument('--top-k', type=int, help='Top-k')
    parser.add_argument('--max-tokens', type=int, help='Max tokens')
    
    args = parser.parse_args()
    
    client = LLMWSEnhancedClient(host=args.host, port=args.port)
    
    # Connect
    if not await client.connect(session_id=args.session):
        sys.exit(1)
    
    try:
        # Update config if parameters provided
        config_updates = {}
        if args.temperature is not None:
            config_updates['temperature'] = args.temperature
        if args.top_p is not None:
            config_updates['top_p'] = args.top_p
        if args.top_k is not None:
            config_updates['top_k'] = args.top_k
        if args.max_tokens is not None:
            config_updates['max_new_tokens'] = args.max_tokens
        
        if config_updates:
            await client.set_config(**config_updates)
        
        # Single prompt or interactive
        if args.prompt:
            image_payload = [{"path": str(Path(p).expanduser())} for p in (args.image or [])]
            await client.inference(
                prompt=args.prompt,
                system=args.system,
                images=image_payload if image_payload else None,
            )
        else:
            await client.interactive_mode()
    
    finally:
        await client.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        sys.exit(1)

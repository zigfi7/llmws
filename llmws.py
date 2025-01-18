#!/usr/bin/python
import os, re, asyncio
import torch
import websockets
from websockets.exceptions import WebSocketException
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import is_flash_attn_2_available

ws_port = 8765
modeldir = 'Phi-3.5-mini-instruct'
#modeldir = 'phi-4'
model_path = os.path.join(os.getcwd(), 'models', modeldir)
total_max_tokens = 128000
model = None
space_token = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def is_flash_attention_supported():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return (major, minor) >= (8, 0)
    return False

use_flash_attention = is_flash_attention_supported() and is_flash_attn_2_available()

if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print(f"Use Flash Attention: {use_flash_attention}")


client_queues = defaultdict(list)
processing_flags = defaultdict(bool)
token_buffers = defaultdict(str)
id_buffers = defaultdict(list)
collecting_tags = defaultdict(bool)

async def load_model():
    global model, tokenizer, generation_config, space_token
    if model is None:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map='auto',
                local_files_only=True,
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if use_flash_attention else None
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                use_fast=True
            )
            generation_config = GenerationConfig(
                max_new_tokens=total_max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )
            space_encoded = tokenizer.encode(" ", add_special_tokens=False)
            space_token = tokenizer.convert_ids_to_tokens(space_encoded[0])
            print(f"Space token: {space_token}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    return model, tokenizer, generation_config

def clean_text(text):     return re.sub(r'\s+', ' ', text).strip()
def untag(tag): return tag.strip('<|>').lower() if '<|' in tag else None
def has_space(token_str):
    if space_token:
        return token_str.startswith(space_token) 
    return None

def flush_id_buffer(websocket):
    if id_buffers[websocket]:
        try:
            decoded = tokenizer.decode(id_buffers[websocket])
            id_buffers[websocket] = []
            return decoded
        except Exception:
            id_buffers[websocket] = []
    return None

async def safe_send(websocket, message):
    try:
        if message:
            await websocket.send(message)
    except WebSocketException:
        pass

async def stream_tokens(model, tokenizer, inputs, generation_config, websocket, remaining_tokens):
    generated = inputs["input_ids"]
    past_key_values = None
    tokens_generated = 0
    last_token_time = asyncio.get_event_loop().time()
    hspc=False

    while generated.size(1) < remaining_tokens:
        try:
            current_time = asyncio.get_event_loop().time()
            if tokens_generated == 0 and current_time - last_token_time > 5:
                raise Exception("Token generation timeout")

            with torch.no_grad():
                outputs = model(
                    input_ids=generated[:, -1:] if past_key_values else generated,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=-1)

                token_str = tokenizer.convert_ids_to_tokens(next_token[0].item())
                current_id = next_token[0].item()
                tag=untag(token_str)
                
                if next_token.item() == tokenizer.eos_token_id or tag and "end" in tag:
                    final_text = flush_id_buffer(websocket)
                    if final_text:
                        await safe_send(websocket, final_text)
                    await websocket.close(code=1000)
                    break

                id_buffers[websocket].append(current_id)
                if has_space(token_str):
                    text = flush_id_buffer(websocket)
                    await safe_send(websocket, text)

                if hspc and token_str.startswith(space_token[0]):
                    text = flush_id_buffer(websocket)
                    text = " " + text
                    hspc=False
                    await safe_send(websocket, text)

                if len(space_token)>1 and token_str.endswith(space_token[0]):
                    hspc=True

                tokens_generated += 1
                last_token_time = asyncio.get_event_loop().time()
            await asyncio.sleep(0)
        except Exception as e:
            print(f"Error during token generation: {str(e)}")
            raise

async def process_queue(websocket):
    while client_queues[websocket]:
        try:
            prompt = clean_text(client_queues[websocket].pop(0))
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            tokens_received = len(inputs["input_ids"][0])
            remaining_tokens = max(generation_config.max_new_tokens - tokens_received, 0)
            await stream_tokens(model, tokenizer, inputs, generation_config, websocket, remaining_tokens)
        except Exception as e:
            print(f"Error processing prompt: {str(e)}")
            break
    processing_flags[websocket] = False
    client_queues[websocket].clear()

async def handle_connection(websocket):
    try:
        async for message in websocket:
            if len(client_queues[websocket]) < 1024:
                client_queues[websocket].append(message)
                if not processing_flags[websocket]:
                    processing_flags[websocket] = True
                    asyncio.create_task(process_queue(websocket))
    except WebSocketException:
        pass
    finally:
        client_queues.pop(websocket, None)
        processing_flags.pop(websocket, None)
        token_buffers.pop(websocket, None)
        id_buffers.pop(websocket, None)
        collecting_tags.pop(websocket, None)
        try:
            await websocket.close()
        except:
            pass

async def main():
    print("Loading model...")
    global model, tokenizer, generation_config
    model, tokenizer, generation_config = await load_model()
    print("Model loaded successfully")

    async with websockets.serve(handle_connection, "0.0.0.0", ws_port, max_size=2**23):
        print(f"WebSocket server running on port {ws_port}")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server shutdown")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
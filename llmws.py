#!/usr/bin/python
import os, re, asyncio
import torch
import websockets
from websockets.exceptions import WebSocketException
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import is_flash_attn_2_available

ws_port = 8765
modeldir = 'Llama-3.1-Tulu-3-8B'
#modeldir = 'Phi-3.5-mini-instruct'

model_path = os.path.join(os.getcwd(), 'models', modeldir)

total_max_tokens = 128000
model = None
space_token = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

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

def parse_input_prompt(full_prompt):
  system_match = re.search(r"<\|system\|>(.*?)<\|end\|>", full_prompt, re.DOTALL)
  user_match = re.search(r"<\|user\|>(.*?)<\|end\|>", full_prompt, re.DOTALL)
  system_prompt = system_match.group(1).strip() if system_match else ""
  user_prompt = user_match.group(1).strip() if user_match else ""
  return system_prompt, user_prompt

def prompt2scheme(system_prompt, user_prompt):
  chat = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_prompt},
  ]
  try:
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
  except Exception as e:
    sys_section = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n" if system_prompt else ""
    return f"[INST]{sys_section}{user_prompt}[/INST]"

client_queues = defaultdict(list)
processing_flags = defaultdict(bool)
id_buffers = defaultdict(list)
sent_length = defaultdict(int)
collecting_tags = defaultdict(bool)

async def load_model():
  global model, tokenizer, generation_config
  if model is None:
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    try:
      attn_implementation = None
      torch_dtype = "auto"
      
      if use_flash_attention:
        torch_dtype = torch.float16
        attn_implementation = "flash_attention_2"
      
      print(f"Loading with dtype: {torch_dtype}, attention implementation: {attn_implementation}")
      
      model = AutoModelForCausalLM.from_pretrained(
          model_path,
          torch_dtype=torch_dtype,
          device_map='auto',
          local_files_only=True,
          trust_remote_code=True,
          attn_implementation=attn_implementation
      )
      
      model_config = model.config
      model_type = getattr(model_config, '_name_or_path', modeldir)
      model_architecture = model_config.__class__.__name__
      model_params = sum(p.numel() for p in model.parameters())
      model_params_in_billions = model_params / 1e9
      
      quant_info = "No quantization"
      if hasattr(model_config, 'quantization_config') and model_config.quantization_config:
          quant_info = f"Quantized: {model_config.quantization_config}"
      elif hasattr(model, 'is_quantized') and model.is_quantized:
          quant_info = "Model is quantized (method unspecified)"
      
      print(f"Model loaded: {modeldir}")
      print(f"Architecture: {model_architecture}")
      print(f"Parameters: {model_params_in_billions:.2f}B")
      print(f"Quantization: {quant_info}")
      print(f"Model dtype: {model.dtype}")
      
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
      print("Model and tokenizer loaded successfully")
    except Exception as e:
      print(f"Error loading model: {str(e)}")
      raise
  return model, tokenizer, generation_config

def clean_text(text): return re.sub(r'\s+', ' ', text).strip()
def untag(tag): return tag.strip('<|>').lower() if '<|' in tag else None

async def safe_send(websocket, message):
  try:
    if message:
      await websocket.send(message)
  except WebSocketException:
    pass

async def stream_tokens(model, tokenizer, inputs, generation_config, websocket, remaining_tokens):
  generated = inputs["input_ids"].to(model.device)  # Ensure inputs are on the correct device
  past_key_values = None
  tokens_generated = 0
  last_token_time = asyncio.get_event_loop().time()

  MULTILINGUAL_SEPARATORS = {
    ' ', '\t', '\n',
    ',', '.', '!', '?', ';', ':',
    '，', '。', '！', '？', '；', '：', '、', '…', '—', '～', '-'
  }

  # Only use autocast if we're using flash attention and on CUDA
  use_autocast = use_flash_attention and torch.cuda.is_available()
  
  async def process_tokens():
    nonlocal generated, past_key_values, tokens_generated, last_token_time
    
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

          current_id = next_token[0].item()
          token_str = tokenizer.convert_ids_to_tokens(current_id)
          tag = untag(token_str)

          if next_token.item() == tokenizer.eos_token_id or (tag and "end" in tag):
            current_text = tokenizer.decode(id_buffers[websocket])
            new_text = current_text[sent_length[websocket]:]
            if new_text:
              await safe_send(websocket, new_text)
              sent_length[websocket] = len(current_text)
            await websocket.close(code=1000)
            break

          id_buffers[websocket].append(current_id)
          current_text = tokenizer.decode(id_buffers[websocket])
          new_text = current_text[sent_length[websocket]:]

          last_separator_pos = -1
          for i in reversed(range(len(new_text))):
            if new_text[i] in MULTILINGUAL_SEPARATORS:
              last_separator_pos = i + 1
              break

          if last_separator_pos != -1:
            to_send = new_text[:last_separator_pos]
            await safe_send(websocket, to_send)
            sent_length[websocket] += len(to_send)

          tokens_generated += 1
          last_token_time = current_time
        await asyncio.sleep(0)
      except Exception as e:
        print(f"Token generation error: {str(e)}")
        raise
  
  try:
    if use_autocast:
      with torch.amp.autocast('cuda', dtype=model.dtype):
        await process_tokens()
    else:
      await process_tokens()
  except Exception as e:
    print(f"Error in token processing: {str(e)}")
    raise

async def process_queue(websocket):
  while client_queues[websocket]:
    try:
      input_prompt = clean_text(client_queues[websocket].pop(0))
      system_prompt, user_prompt = parse_input_prompt(input_prompt)
      prompt = prompt2scheme(system_prompt, user_prompt)
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
    id_buffers.pop(websocket, None)
    sent_length.pop(websocket, None)
    collecting_tags.pop(websocket, None)
    try:
      await websocket.close()
    except:
      pass

async def main():
  print("Loading model:", modeldir)
  global model, tokenizer, generation_config
  model, tokenizer, generation_config = await load_model()
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

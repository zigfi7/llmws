# LLMWS Status

Last update (UTC): 2026-02-17 19:31:59
Visibility: public-safe tracker (no host/user identifiers)

## Product Direction

- Keep `llmws` lightweight and portable across different machines and architectures.
- Keep environment strategy flexible (`mamba` + fallback options) for heterogeneous hosts.
- Focus execution path now on `txt2txt` and `img+txt2txt`.
- Keep API shape extensible toward `any in -> any out` where the selected model supports it.
- Plan remote training workflows as optional modules (without bloating inference wrapper).

## Current State

- Git branch: `main`
- Runtime code baseline (multimodal patch): `f8601ae`
- Runtime host/service details are tracked only in `STATUS.private.md`.
- Current validated inference paths:
  - `txt2txt` works end-to-end
  - `img+txt2txt` works end-to-end (with temporary compatibility guards)
- Current validated training/snapshot paths:
  - `train_status` works
  - `save_snapshot` writes `safetensors`
  - `train` works on tiny JSONL smoke jobs

## Functional Audit (2026-02-17)

- Environment management:
  - `start.sh` supports `auto|venv|mamba` backend selection.
  - `setup.sh` and `start.sh` bash syntax checks pass.
  - Startup help and backend priority flow are working.
- Training functions:
  - Server API now supports `train` and `train_status`.
  - JSONL dataset training works with progress events (`train_started`, `train_progress`, `train_done`).
  - Added dedicated client: `llmws_train_client.py`.
  - Added training-loss fallback to explicit CE over logits when model-provided loss is detached.
- Snapshot saving (`safetensors`):
  - Server API now supports `save_snapshot`.
  - Snapshot saving is active via `safetensors` (`model.safetensors`) plus config/tokenizer artifacts.
  - Training can save final checkpoint and periodic checkpoints (`save_every`).
- Multimodal implementation:
  - Loader fallback to VLM path is implemented.
  - Image parsing supports `path`, `base64`, `data_url`, `url`.
  - Runtime supports `txt2txt` and `img+txt2txt` request flow.
  - Current stack still relies on temporary multimodal safety guards noted below.

## Environment Snapshot (tested stack)

- `torch`: `2.10.0+cu130`
- `transformers`: `5.1.0`
- `websockets`: `16.0`
- `requests`: `2.32.5`
- `einops`: `0.8.2`

## Recent Work

1. Added Molmo2-capable loading path with fallback from `AutoModelForCausalLM` to `AutoModelForImageTextToText`.
2. Added multimodal image ingestion (`path`, `base64`, `data_url`, `url`) and unified prompt parsing.
3. Added multimodal inference path plus client-side `--image` support.
4. Added training + snapshot functionality back into current server API.
5. Added `llmws_train_client.py` for training/status/snapshot operations.
6. Added operational tracker workflow (public + private split).
7. Stabilized runtime against known CUDA/runtime crashes on current stack.
8. Revalidated full runtime on `dl8` after deployment sync; restored model directory and rechecked all critical paths.

## Recovery Notes (2026-02-17)

- A deployment sync attempt removed runtime model artifacts on target host.
- Model payload was restored from upstream (`allenai/Molmo2-8B`) into runtime `models/`.
- Post-recovery smoke checks passed for:
  - `txt2txt`
  - `img+txt2txt`
  - `train_status`
  - `save_snapshot`
  - `train` (1-step smoke)

## Temporary Compatibility Logic (To Remove)

- `rope_type='default'` -> `'linear'` runtime patch for current `transformers` compatibility.
- Lenient `ProcessorMixin` init wrapper for remote processor kwargs mismatch.
- Multimodal safety overrides:
  - force `do_sample=False`
  - force `repetition_penalty=1.0`

## Known Issues

- Generation quality with current Molmo2 runtime stack is inconsistent.
- Runtime relies on temporary compatibility guards that should be eliminated by proper version alignment.
- Deployment process still needs full standardization for multi-host rollouts.

## Next Steps (No-Workaround Direction)

1. Pin a Molmo2-compatible `transformers` version and rebuild env.
2. Re-validate `txt2txt`, `img+txt2txt`, and `train+snapshot` without runtime patches.
3. Remove temporary compatibility logic once stack is stable.
4. Add standardized multi-host deploy + smoke-test routine.
5. Extend training pipeline with dataset chunking/streaming for larger corpora.

## Quick Checks (Template)

```bash
# service status
ssh <target_host> 'systemctl is-active llmws.service'

# websocket listener
ssh <target_host> 'ss -ltnp | grep ":8765 "'

# txt2txt smoke
ssh <target_host> 'cd /path/to/llmws && <python_env>/python llm.py --host 127.0.0.1 --port 8765 --max-tokens 8 -p "OK"'

# img+txt2txt smoke
ssh <target_host> 'cd /path/to/llmws && <python_env>/python llm.py --host 127.0.0.1 --port 8765 --max-tokens 16 -p "Describe image" --image /tmp/test.png'
```

# LLMWS Status

Last update (UTC): 2026-02-17 18:36:30
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
4. Added operational tracker workflow (public + private split).
5. Stabilized runtime against known CUDA/runtime crashes on current stack.

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
2. Re-validate `txt2txt` and `img+txt2txt` without runtime patches.
3. Remove temporary compatibility logic once stack is stable.
4. Add standardized multi-host deploy + smoke-test routine.
5. Define minimal remote-training interface (`submit`, `monitor`, `collect`) as separate component.

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

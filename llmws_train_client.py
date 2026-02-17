#!/usr/bin/env python3
"""
LLMWS training client for current JSON protocol.
Supports: start train job, resume by request_id, status query, snapshot.
"""
import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Optional, List

import websockets


class LLMWSTrainClient:
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.ws = None
        self.uri = None

    async def connect(self) -> bool:
        last_error = None
        for target in self.servers:
            uri = target if target.startswith("ws://") or target.startswith("wss://") else f"ws://{target}"
            try:
                self.ws = await websockets.connect(
                    uri,
                    max_size=2**26,
                    ping_interval=30,
                    ping_timeout=10,
                )
                self.uri = uri
                await self.ws.send(json.dumps({}))
                welcome = json.loads(await self.ws.recv())
                if welcome.get("type") == "welcome":
                    print(f"Connected: {uri}")
                    print(f"Session: {welcome.get('session_id')}")
                    print(f"Model: {welcome.get('model')}")
                    return True
                print(f"Unexpected welcome from {uri}: {welcome}")
                await self.ws.close()
                self.ws = None
            except Exception as err:
                last_error = err
                print(f"Failed {uri}: {err}")

        print(f"No reachable server: {last_error}")
        return False

    async def close(self):
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "ack"}))
                await asyncio.wait_for(self.ws.recv(), timeout=2.0)
            except Exception:
                pass
            await self.ws.close()

    async def get_train_status(self):
        await self.ws.send(json.dumps({"type": "train_status"}))
        data = json.loads(await self.ws.recv())
        if data.get("type") != "train_status":
            raise RuntimeError(f"Unexpected response: {data}")
        return data.get("training", {})

    async def save_snapshot(self, name: Optional[str] = None):
        payload = {"type": "save_snapshot"}
        if name:
            payload["name"] = name
        await self.ws.send(json.dumps(payload))
        data = json.loads(await self.ws.recv())
        if data.get("type") == "snapshot_saved":
            return data.get("path")
        raise RuntimeError(data.get("message", str(data)))

    async def train(
        self,
        request_id: str,
        dataset_text: Optional[str],
        resume: bool,
        save_checkpoint: bool,
        checkpoint_name: Optional[str],
        config: dict,
    ):
        payload = {
            "type": "train",
            "request_id": request_id,
            "resume": bool(resume),
            "save_checkpoint": bool(save_checkpoint),
            "checkpoint_name": checkpoint_name,
            "config": config,
        }
        if dataset_text is not None:
            payload["dataset"] = dataset_text

        await self.ws.send(json.dumps(payload))

        async for raw in self.ws:
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "train_waiting":
                print(f"[{request_id}] waiting for train lock")
                continue

            if msg_type == "train_started":
                print(f"[{request_id}] started")
                print(f"config={data.get('config')}")
                continue

            if msg_type == "train_progress":
                step = data.get("step")
                max_steps = data.get("max_steps")
                loss = data.get("loss")
                print(f"[{request_id}] step {step}/{max_steps} loss={loss}")
                continue

            if msg_type == "train_checkpoint":
                print(f"[{request_id}] checkpoint: {data.get('path')}")
                continue

            if msg_type == "train_done":
                print(f"[{request_id}] done steps={data.get('steps')} checkpoint={data.get('checkpoint')}")
                return data

            if msg_type == "error":
                raise RuntimeError(data.get("message", str(data)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLMWS training client")
    parser.add_argument("-s", "--server", action="append", required=True, help="host:port or ws:// URI (repeatable)")

    parser.add_argument("-d", "--dataset", help="Path to JSONL dataset")
    parser.add_argument("--resume", action="store_true", help="Resume using server-cached dataset for request_id")
    parser.add_argument("--request-id", default=None, help="Stable request ID for resume flow")

    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--grad-accum", type=int, default=1)

    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--no-save", action="store_true", help="Do not save final checkpoint")

    parser.add_argument("--status", action="store_true", help="Only query train status")
    parser.add_argument("--snapshot", nargs="?", const="", default=None, help="Only save snapshot (optional name)")
    return parser


async def async_main():
    parser = build_parser()
    args = parser.parse_args()

    request_id = args.request_id or str(uuid.uuid4())

    dataset_text = None
    if args.dataset:
        dataset_path = Path(args.dataset).expanduser()
        if not dataset_path.exists():
            print(f"Dataset not found: {dataset_path}")
            return 1
        dataset_text = dataset_path.read_text(encoding="utf-8")
    elif not args.resume and not args.status and args.snapshot is None:
        print("Provide --dataset or use --resume")
        return 1

    train_cfg = {
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_seq_length": args.max_seq_length,
        "log_every": args.log_every,
        "save_every": args.save_every,
        "gradient_accumulation": args.grad_accum,
    }

    client = LLMWSTrainClient(args.server)
    if not await client.connect():
        return 1

    try:
        if args.status:
            status = await client.get_train_status()
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return 0

        if args.snapshot is not None:
            snapshot_name = args.snapshot.strip() or None
            path = await client.save_snapshot(snapshot_name)
            print(f"snapshot_saved={path}")
            return 0

        await client.train(
            request_id=request_id,
            dataset_text=dataset_text,
            resume=args.resume,
            save_checkpoint=not args.no_save,
            checkpoint_name=args.checkpoint_name,
            config=train_cfg,
        )
        return 0
    except Exception as err:
        print(f"Training failed: {err}")
        return 1
    finally:
        await client.close()


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(async_main()))
    except KeyboardInterrupt:
        print("Interrupted")
        raise SystemExit(130)

#!/usr/bin/env python3
"""Generate story artifacts against a vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import string
import time
import urllib.error
import urllib.request
from pathlib import Path


GENRES = [
    "weird maritime folklore",
    "near-future desert logistics",
    "botanical espionage",
    "post-collapse library politics",
    "orbital salvage drama",
    "small-town weather mystery",
    "underground transit myth",
    "arctic engineering fable",
    "clockmaker courtroom intrigue",
    "deep-sea agricultural expedition",
]

OBJECTS = [
    "a brass seed vault",
    "a map printed on dissolving salt",
    "a train schedule with impossible stations",
    "a ceramic bird that records lies",
    "a weather balloon full of old letters",
    "a broken compass that points to debts",
    "an elevator key from a demolished hotel",
    "a lantern that burns cold",
    "a glacier core wrapped in red thread",
    "a ledger written in mirrored ink",
]

TONES = [
    "tense and precise",
    "dryly funny",
    "lush and strange",
    "plainspoken but ominous",
    "fast-moving and cinematic",
    "melancholy and exact",
]


def request_json(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_existing(manifest: Path) -> tuple[int, int]:
    if not manifest.exists():
        return 0, 0

    total_tokens = 0
    count = 0
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("ok"):
                total_tokens += int(record.get("completion_tokens", 0))
                count += 1
    return total_tokens, count


def make_prompt(index: int) -> tuple[str, dict]:
    seed = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(12))
    spec = {
        "index": index,
        "seed": seed,
        "genre": random.choice(GENRES),
        "object": random.choice(OBJECTS),
        "tone": random.choice(TONES),
    }
    prompt = (
        "Write a complete standalone short story of at least 5,000 visible characters. "
        "Do not include analysis, notes, markdown headings, or a title. "
        "Make it random in nature while still coherent and complete. "
        "End with a complete sentence.\n\n"
        f"Story seed: {spec['seed']}\n"
        f"Genre pressure: {spec['genre']}\n"
        f"Required object: {spec['object']}\n"
        f"Tone: {spec['tone']}\n"
        f"Run index: {index}\n"
    )
    return prompt, spec


def generate_one(
    *,
    base_url: str,
    model: str,
    index: int,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> dict:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    prompt, spec = make_prompt(index)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    last_error = None
    started = time.time()
    for attempt in range(1, retries + 2):
        try:
            response = request_json(url, payload, timeout)
            choice = response["choices"][0]
            content = choice["message"].get("content") or ""
            usage = response.get("usage") or {}
            return {
                "ok": True,
                "index": index,
                "spec": spec,
                "content": content,
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
                "chars": len(content),
                "elapsed_seconds": time.time() - started,
                "attempt": attempt,
                "created_at": time.time(),
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            last_error = repr(exc)
            time.sleep(min(30, 2 * attempt))

    return {
        "ok": False,
        "index": index,
        "spec": spec,
        "error": last_error,
        "completion_tokens": 0,
        "chars": 0,
        "elapsed_seconds": time.time() - started,
        "created_at": time.time(),
    }


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-completion-tokens", type=int, default=1_000_000)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    stories_dir = out_dir / "stories"
    stories_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    summary = out_dir / "summary.json"

    total_tokens, completed = load_existing(manifest)
    next_index = completed + 1
    start_time = time.time()
    print(
        f"starting total_completion_tokens={total_tokens} completed={completed} "
        f"target={args.target_completion_tokens} out_dir={out_dir}",
        flush=True,
    )

    futures: set[concurrent.futures.Future] = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        while total_tokens < args.target_completion_tokens or futures:
            while total_tokens < args.target_completion_tokens and len(futures) < args.concurrency:
                futures.add(
                    executor.submit(
                        generate_one,
                        base_url=args.base_url,
                        model=args.model,
                        index=next_index,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        retries=args.retries,
                    )
                )
                next_index += 1

            done, futures = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                record = fut.result()
                if record.get("ok"):
                    story_path = stories_dir / f"story_{record['index']:06d}.txt"
                    story_path.write_text(record["content"], encoding="utf-8")
                    record["story_path"] = str(story_path)
                    total_tokens += int(record["completion_tokens"])
                    completed += 1
                append_jsonl(manifest, record)

                elapsed = max(0.001, time.time() - start_time)
                rate = total_tokens / elapsed
                print(
                    "progress "
                    f"completed={completed} total_completion_tokens={total_tokens} "
                    f"last_tokens={record.get('completion_tokens', 0)} "
                    f"last_chars={record.get('chars', 0)} "
                    f"ok={record.get('ok')} tokens_per_sec={rate:.2f}",
                    flush=True,
                )

        summary_record = {
            "target_completion_tokens": args.target_completion_tokens,
            "completion_tokens": total_tokens,
            "stories_completed": completed,
            "elapsed_seconds": time.time() - start_time,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "base_url": args.base_url,
            "model": args.model,
            "finished_at": time.time(),
        }
        summary.write_text(json.dumps(summary_record, indent=2), encoding="utf-8")
        print("summary " + json.dumps(summary_record, sort_keys=True), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

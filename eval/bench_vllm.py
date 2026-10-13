#!/usr/bin/env python3
"""vLLM benchmark mirroring eval/bench.py methodology.

Same sweep grid (batch x prompt_len x new_tokens), same metrics:
  TTFT p50/p99, ITL p50/p99, TPOT mean, throughput tok/s.
Greedy decoding, seed fixed. Uses /v1/completions (no chat template needed --
the local model is a base model and bench.py uses raw prompts too).

Per-request timing is measured from streaming chunks: first chunk => TTFT,
subsequent inter-chunk gaps => ITL (matches engine.py's token_times_ms).
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import requests
from transformers import AutoTokenizer

MODEL_DIR = Path.home() / "Desktop/sds/llm-inference-engine/python/models/llama-3.2-1b"
BASE = "http://localhost:8000/v1/completions"
MODEL_NAME = "llama-3.2-1b"

tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))

WORDS = (
    "time year people way day man thing woman life child world school state family "
    "student group country problem hand part place case week company system program "
    "question work government number night point home water room mother area money "
    "story fact month lot right study book eye job word business issue side kind head "
    "house service friend father power hour game line end member law car city community "
    "name president team minute idea body information parent face level office door "
    "health person art war history party result change morning reason research girl "
    "moment air teacher force education season player market report street while"
).split()


def text_of_token_length(target, rng):
    """Real text encoding to exactly `target` tokens (matches bench.py)."""
    if target < 2:
        return None
    words = [rng.choice(WORDS) for _ in range(target)]
    for _ in range(60):
        text = " ".join(words)
        length = len(tok(text)["input_ids"])
        if length == target:
            return text
        if length < target:
            words.extend(rng.choice(WORDS) for _ in range(target - length))
        else:
            trim = min(length - target, len(words) - 1)
            words = words[: len(words) - trim]
    return None


def percentiles(values, points=(50, 90, 99)):
    if not values:
        return {f"p{p}": 0.0 for p in points}
    arr = np.array(values, dtype=float)
    return {f"p{p}": float(np.percentile(arr, p)) for p in points}


def run_one_streaming(text, max_tokens, rid):
    """Stream a completion; return (ttft_ms, token_times_ms, gen_count)."""
    t0 = time.perf_counter()
    times = []
    gen = 0
    with requests.post(BASE, json={
        "model": MODEL_NAME, "prompt": text, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True, "seed": 0,
    }, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[len(b"data: "):]
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage") is None and not chunk["choices"]:
                continue
            if chunk["choices"][0].get("text") is None:
                continue
            times.append((time.perf_counter() - t0) * 1000)
            gen += 1
    ttft = times[0] if times else 0.0
    return ttft, times, gen


def run_config(batch, prompt_len, new_tokens, reps, rng):
    ttfts, itls = [], []
    wall_ms = []
    prompt_tokens = []
    generated_total = 0

    for _ in range(reps):
        prompts = []
        for i in range(batch):
            text = text_of_token_length(prompt_len, rng)
            if text is None:
                sys.exit(f"could not build a prompt of exactly {prompt_len} tokens")
            prompts.append(text)

        wall0 = time.perf_counter()
        for text in prompts:
            ttft, times, gen = run_one_streaming(text, new_tokens, "r")
            ttfts.append(ttft)
            itls.extend(b - a for a, b in zip(times, times[1:]))
            prompt_tokens.append(prompt_len)
            generated_total += gen
        wall_ms.append((time.perf_counter() - wall0) * 1000)

    total_wall_s = sum(wall_ms) / 1000.0
    throughput = generated_total / total_wall_s if total_wall_s > 0 else 0.0
    tpot = statistics.mean(itls) if itls else 0.0

    return {
        "batch": batch,
        "prompt_len": prompt_len,
        "mean_prompt_tokens": statistics.mean(prompt_tokens) if prompt_tokens else prompt_len,
        "new_tokens": new_tokens,
        "reps": reps,
        "ttft_ms": percentiles(ttfts),
        "itl_ms": percentiles(itls),
        "tpot_ms_mean": tpot,
        "throughput_tok_s": throughput,
        # note: no MBU/roofline -- vLLM does continuous batching + paged KV,
        # bytes-per-step accounting differs fundamentally from a static engine.
        "engine": "vllm-0.31.0",
        "device": {"name": "NVIDIA GeForce RTX 4050 Laptop GPU"},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="1,2,4")
    parser.add_argument("--prompt-len", default="32,128,512")
    parser.add_argument("--new-tokens", default="64")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="write full results here")
    args = parser.parse_args()

    batches = [int(x) for x in args.batch.split(",")]
    prompt_lens = [int(x) for x in args.prompt_len.split(",")]
    new_tokens = [int(x) for x in args.new_tokens.split(",")]

    rng = random.Random(args.seed)
    results = []

    header = (f"{'batch':>5} {'plen':>5} {'gen':>5} "
              f"{'TTFT p50':>9} {'p99':>8} {'ITL p50':>8} {'p99':>8} "
              f"{'TPOT':>7} {'tok/s':>8}")
    print(header)
    print("-" * len(header))

    for batch in batches:
        for prompt_len in prompt_lens:
            for gen in new_tokens:
                result = run_config(batch, prompt_len, gen, args.reps, rng)
                results.append(result)
                print(f"{batch:>5} {prompt_len:>5} {gen:>5} "
                      f"{result['ttft_ms']['p50']:>9.1f} {result['ttft_ms']['p99']:>8.1f} "
                      f"{result['itl_ms']['p50']:>8.2f} {result['itl_ms']['p99']:>8.2f} "
                      f"{result['tpot_ms_mean']:>7.2f} {result['throughput_tok_s']:>8.1f}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

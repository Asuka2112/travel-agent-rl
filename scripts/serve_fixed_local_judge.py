#!/usr/bin/env python3
"""Serve the immutable Base-model Judge on CPU for GRPO reward calls."""
from __future__ import annotations

import argparse
import json
import re
import threading
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from travel_agent_rl.filtering.judge import JudgeScores
from travel_agent_rl.sft.data import sha256_file

app = FastAPI(title="Travel Agent Fixed Local Judge")
MODEL: Any = None
TOKENIZER: Any = None
PROMPT = ""
MODEL_PATH = ""
GENERATION_LOCK = threading.Lock()


class JudgeRequest(BaseModel):
    trajectories: list[list[dict[str, Any]]] = Field(min_length=1)


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    first, last = stripped.find("{"), stripped.rfind("}")
    if first < 0 or last <= first:
        raise ValueError("no JSON object")
    payload = json.loads(stripped[first : last + 1])
    if not isinstance(payload, dict):
        raise TypeError("Judge output is not an object")
    return payload


def compact(messages: list[dict[str, Any]]) -> str:
    rows = []
    for message in messages:
        rows.append(
            {
                "role": message.get("role"),
                "content": str(message.get("content", ""))[:1800],
            }
        )
    return json.dumps({"messages": rows}, ensure_ascii=False)[:24000]


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if MODEL is not None else "loading",
        "model": MODEL_PATH,
        "judge_prompt_sha256": sha256_file(Path("prompts/judge/v1.txt")),
        "device": "cpu",
        "policy_model": False,
    }


@app.post("/judge")
def judge(request: JudgeRequest) -> dict[str, Any]:
    conversations = [
        [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": compact(messages)},
        ]
        for messages in request.trajectories
    ]
    rendered = [
        TOKENIZER.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for conversation in conversations
    ]
    batch = TOKENIZER(rendered, return_tensors="pt", padding=True, truncation=True, max_length=30000)
    with GENERATION_LOCK, torch.inference_mode():
        outputs = MODEL.generate(
            **batch,
            do_sample=False,
            max_new_tokens=320,
            pad_token_id=TOKENIZER.eos_token_id,
        )
    scores = []
    prompt_length = batch["input_ids"].shape[1]
    for output in outputs:
        raw = TOKENIZER.decode(output[prompt_length:], skip_special_tokens=True)
        try:
            payload = extract_json(raw)
            dimensions = JudgeScores.model_validate(payload["scores"])
            scores.append(
                {
                    "score": dimensions.mean / 5.0,
                    "dimensions": dimensions.model_dump(),
                    "failed": False,
                }
            )
        except Exception as exc:
            scores.append(
                {
                    "score": 0.0,
                    "failed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {"scores": scores}


def main() -> None:
    global MODEL, MODEL_PATH, PROMPT, TOKENIZER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--torch-threads", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(min(8, args.torch_threads))
    MODEL_PATH = str(args.model)
    PROMPT = Path("prompts/judge/v1.txt").read_text(encoding="utf-8")
    TOKENIZER = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    TOKENIZER.padding_side = "left"
    if TOKENIZER.pad_token_id is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token
    MODEL = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cpu",
    ).eval()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

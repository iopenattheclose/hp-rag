"""Local LLM wrapper (MLX).  ONE function matters: LLM.generate(messages) -> str.

If you already have your own MLX script, replace the body of generate() with your call;
nothing else in the project depends on how this works.

Env: LLM_MODEL = local folder or HF repo id of your MLX Qwen model.

Loop protection (the repeated "the one who later became..." output): chat template applied,
sampling instead of greedy decoding, a mild repetition penalty and a hard max_tokens.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

import re


class LLM:
    def __init__(self, model: str | None = None, temp: float = 0.3, top_p: float = 0.9,
                 repetition_penalty: float = 1.05, max_tokens: int = 400):
        model = model or os.getenv("MODEL_PATH")
        if not model:
            raise SystemExit("Set LLM_MODEL (path or repo id of your MLX model) or pass --llm-model.")
        from mlx_lm import load
        self.model, self.tok = load(model)
        self.temp, self.top_p, self.rep, self.max_tokens = temp, top_p, repetition_penalty, max_tokens
        print(f"[llm] loaded {model}")

    def generate(self, messages: list[dict]) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
        try:    # Qwen3 templates accept enable_thinking; other templates ignore or reject it
            prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                                  tokenize=False, enable_thinking=False)
        except TypeError:
            prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        out = generate(
            self.model, self.tok, prompt=prompt, max_tokens=self.max_tokens,
            sampler=make_sampler(temp=self.temp, top_p=self.top_p),
            logits_processors=make_logits_processors(repetition_penalty=self.rep),
        )
        return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
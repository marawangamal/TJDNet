#!/usr/bin/env python3
"""
Ultra-light chat REPL.
- Handles both <CR> and <CR><LF> line endings.
- Shows a spinner while the model is thinking.
"""

from __future__ import annotations
import argparse
import sys
import threading
import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------- #
# I/O helper that accepts CR or LF                                            #
# --------------------------------------------------------------------------- #
def read_line(prompt: str = "") -> str:
    """Return a line of text, accepting either '\\n' or '\\r' as terminator."""
    sys.stdout.write(prompt)
    sys.stdout.flush()

    buf: List[str] = []
    while True:
        ch = sys.stdin.read(1)
        if ch in ("\n", "\r", ""):  # end of line / stream
            # If we got CR and the next char is LF, swallow it
            if ch == "\r":
                sys.stdin.peek = getattr(sys.stdin, "peek", None)  # type: ignore[attr-defined]
                if sys.stdin.peek and sys.stdin.peek(1) == "\n":
                    sys.stdin.read(1)
            break
        buf.append(ch)
    return "".join(buf)


# --------------------------------------------------------------------------- #
# Model helpers                                                               #
# --------------------------------------------------------------------------- #
def load(model_id: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.pad_token = tok.pad_token or tok.eos_token or tok.unk_token or "<|pad|>"

    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto" if device.type == "cuda" else None,
    ).to(device)
    mdl.eval()
    return tok, mdl


def build_prompt(turns: List[dict], tok) -> torch.Tensor:
    if getattr(tok, "chat_template", None):
        txt = tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=True)
    else:  # raw LM like GPT-2 → last user msg only
        txt = turns[-1]["msg"] + "\n"
    return tok(txt, return_tensors="pt").input_ids


class Spinner(threading.Thread):
    glyphs = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label="🧠 Thinking"):
        super().__init__(daemon=True)
        self.label = label
        self._run = True

    def run(self):
        i = 0
        while self._run:
            g = self.glyphs[i % len(self.glyphs)]
            print(f"\r{self.label} {g}", end="", flush=True)
            time.sleep(0.12)
            i += 1
        print("\r" + " " * (len(self.label) + 2) + "\r", end="", flush=True)

    def stop(self):
        self._run = False


# --------------------------------------------------------------------------- #
# Main loop                                                                   #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF Hub ID or local path")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, mdl = load(args.model, device)
    gen_cfg = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    turns: List[dict] = []
    print("💬  Chat ready — Ctrl-C or /exit to quit\n")

    try:
        while True:
            user = read_line("You: ").strip()
            if user.lower() in {"/exit", "/quit"}:
                break
            if not user:
                continue

            turns.append({"role": "user", "msg": user})
            input_ids = build_prompt(turns, tok).to(device)

            spin = Spinner()
            spin.start()
            with torch.inference_mode():
                output = mdl.generate(input_ids, **gen_cfg)
            spin.stop()
            spin.join()

            reply_ids = output[0, input_ids.shape[-1] :]
            reply = tok.decode(reply_ids, skip_special_tokens=True).strip()
            print(f"Bot: {reply}\n", flush=True)
            turns.append({"role": "assistant", "msg": reply})
    except KeyboardInterrupt:
        pass

    print("👋  Bye.", flush=True)


if __name__ == "__main__":
    # Windows & some containers echo CR literally; try to translate
    try:
        import os, subprocess

        # stty is POSIX; ignore errors on Windows
        subprocess.call(
            ["stty", "icrnl"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        pass

    main()

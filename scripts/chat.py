#!/usr/bin/env python3
"""
Tiny chat REPL (ANSI colours, CR/LF-agnostic).

$ python chat.py --model mistralai/Mistral-7B-Instruct-v0.3        # single GPU
$ python chat.py --model big-model --multi-gpu                     # sharded
"""

from __future__ import annotations
import argparse, sys, threading, time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ANSI = {"user": "\033[1;32m", "bot": "\033[1;36m", "dim": "\033[2m", "rst": "\033[0m"}
MAX_HISTORY = 20  # trim old turns so the prompt stays small


# ---------------------- I/O helper ----------------------
def read_line(prompt: str = "") -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    buf: List[str] = []
    while (ch := sys.stdin.read(1)) not in ("\n", "\r", ""):
        buf.append(ch)
    return "".join(buf)


# ---------------------- model utils ----------------------
def load(model_id: str, single_gpu: bool):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.pad_token = tok.pad_token or tok.eos_token or tok.unk_token or "<|pad|>"
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=None if single_gpu else "auto",
    )
    if single_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mdl.to(device)
    else:
        device = torch.device("cpu")  # prompt stays on CPU, Accelerate moves it
    mdl.eval()
    return tok, mdl, device


def prompt(turns: List[dict], tok) -> torch.Tensor:
    if getattr(tok, "chat_template", None):
        chat_turns = [
            {"role": t["role"], "content": t["msg"]}  # ★ adapt key name
            for t in turns[-MAX_HISTORY:]
        ]
        txt = tok.apply_chat_template(
            chat_turns, tokenize=False, add_generation_prompt=True
        )
    else:
        txt = turns[-1]["msg"] + "\n"
    return tok(txt, return_tensors="pt").input_ids


class Spinner(threading.Thread):
    glyphs = "⠋⠙⠚⠞⠴⠦"

    def __init__(self, lbl="🧠 thinking"):
        super().__init__(daemon=True)
        self.lbl, self.run_ = lbl, True

    def run(self):
        i = 0
        while self.run_:
            g = self.glyphs[i % len(self.glyphs)]
            print(f"\r{ANSI['dim']}{self.lbl} {g}{ANSI['rst']}", end="", flush=True)
            time.sleep(0.15)
            i += 1
        print("\r" + " " * (len(self.lbl) + 4) + "\r", end="", flush=True)

    def stop(self):
        self.run_ = False


# ---------------------- main ----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--multi-gpu",
        action="store_true",
        help="shard the model across all visible GPUs (default off)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    tok, mdl, device = load(args.model, single_gpu=not args.multi_gpu)
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
    print(f"{ANSI['bot']}💬  Chat ready — Ctrl-C or /exit to quit{ANSI['rst']}\n")

    try:
        while True:
            user = read_line(f"{ANSI['user']}You:{ANSI['rst']} ").strip()
            if user.lower() in {"/exit", "/quit"}:
                break
            if not user:
                continue
            turns.append({"role": "user", "msg": user})

            ids = prompt(turns, tok)
            if not args.multi_gpu:
                ids = ids.to(device)

            spin = Spinner()
            spin.start()
            with torch.inference_mode():
                out = mdl.generate(ids, **gen_cfg)
            spin.stop()
            spin.join()

            reply = tok.decode(
                out[0, ids.shape[-1] :], skip_special_tokens=True
            ).strip()
            print(f"{ANSI['bot']}Bot:{ANSI['rst']} {reply}\n")
            turns.append({"role": "assistant", "msg": reply})
    except KeyboardInterrupt:
        pass
    print(f"{ANSI['bot']}👋  Bye.{ANSI['rst']}")


if __name__ == "__main__":
    main()

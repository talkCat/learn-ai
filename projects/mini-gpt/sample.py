"""Sample text from a trained Mini GPT checkpoint."""

from __future__ import annotations

import argparse

import torch

from model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/mini_gpt.pt")
    parser.add_argument("--prompt", default="To be")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    stoi = checkpoint["stoi"]
    itos = {int(k): v for k, v in checkpoint["itos"].items()}
    config = GPTConfig(**checkpoint["config"])

    model = GPT(config).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    unknown_chars = sorted(set(args.prompt) - set(stoi))
    if unknown_chars:
        raise ValueError(f"Prompt contains chars not in training vocab: {unknown_chars}")

    start_ids = [stoi[ch] for ch in args.prompt]
    idx = torch.tensor([start_ids], dtype=torch.long, device=args.device)
    generated = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()

    print("".join(itos[i] for i in generated))


if __name__ == "__main__":
    main()

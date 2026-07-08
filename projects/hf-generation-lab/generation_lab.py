"""A small Hugging Face generation experiment lab."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--prompt", default="A software engineer learns transformers by")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runs-dir", default="runs")
    return parser.parse_args()


def next_token_candidates(model, tokenizer, prompt: str, top_k: int):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, top_k)
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "probability": float(prob),
        }
        for rank, (token_id, prob) in enumerate(zip(indices, values), start=1)
    ]


def save_record(record: dict, runs_dir: Path) -> tuple[Path, Path]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = runs_dir / f"{stamp}.json"
    md_path = runs_dir / f"{stamp}.md"

    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(record), encoding="utf-8")
    return json_path, md_path


def to_markdown(record: dict) -> str:
    candidates = "\n".join(
        f"- {item['rank']}. `{item['token']}` id={item['token_id']} "
        f"prob={item['probability']:.6f}"
        for item in record["top_next_tokens"]
    )
    tokens = ", ".join(f"`{token}`" for token in record["tokens"])
    return f"""# Generation Lab Run

## Config

- model: `{record['model']}`
- max_new_tokens: `{record['max_new_tokens']}`
- do_sample: `{record['do_sample']}`
- temperature: `{record['temperature']}`
- top_p: `{record['top_p']}`

## Prompt

```text
{record['prompt']}
```

## Tokens

{tokens}

## Generated Text

```text
{record['generated_text']}
```

## Top Next Tokens

{candidates}
"""


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    inputs = tokenizer(args.prompt, return_tensors="pt")
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})

    output_ids = model.generate(**inputs, **generation_kwargs)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    token_ids = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    top_next_tokens = next_token_candidates(model, tokenizer, args.prompt, args.top_k)

    record = {
        "model": args.model,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "input_ids": token_ids,
        "tokens": tokens,
        "generated_text": generated_text,
        "top_next_tokens": top_next_tokens,
    }

    json_path, md_path = save_record(record, Path(args.runs_dir))

    print("== generated text ==")
    print(generated_text)
    print("\n== tokens ==")
    print(json.dumps(tokens, ensure_ascii=True))
    print("\n== top next tokens ==")
    for item in top_next_tokens:
        print(
            f"{item['rank']:02d}. {ascii(item['token'])} "
            f"id={item['token_id']} prob={item['probability']:.6f}"
        )
    print(f"\nsaved: {json_path}")
    print(f"saved: {md_path}")


if __name__ == "__main__":
    main()

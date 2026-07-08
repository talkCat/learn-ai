"""Inspect how text becomes tokens, token ids, and attention masks."""

import json

from transformers import AutoTokenizer


TEXTS = [
    "Hello, transformer!",
    "我正在学习大模型。",
    "tokenization is not the same as splitting words.",
    "supercalifragilisticexpialidocious",
]


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    for text in TEXTS:
        encoded = tokenizer(text, return_attention_mask=True)
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

        print("\n== text ==")
        print(text)
        print("tokens:", json.dumps(tokens, ensure_ascii=True))
        print("input_ids:", encoded["input_ids"])
        print("attention_mask:", encoded["attention_mask"])
        print("decoded:", tokenizer.decode(encoded["input_ids"]))


if __name__ == "__main__":
    main()

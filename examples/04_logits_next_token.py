"""Print top-k next-token candidates from raw logits."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    model_name = "sshleifer/tiny-gpt2"
    prompt = "The next token is"
    top_k = 10

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[0, -1, :]
    probabilities = torch.softmax(next_token_logits, dim=-1)
    values, indices = torch.topk(probabilities, top_k)

    print(f"prompt: {prompt!r}")
    print("\nTop next-token candidates:")
    for rank, (token_id, prob) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        token = tokenizer.decode([token_id])
        print(f"{rank:02d}. id={token_id:<6} prob={prob:.6f} token={ascii(token)}")


if __name__ == "__main__":
    main()

"""Compare common generation parameters on a tiny causal LM."""

from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_case(model, tokenizer, prompt: str, **kwargs) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=35,
        pad_token_id=tokenizer.eos_token_id,
        **kwargs,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main() -> None:
    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "When a language model predicts the next token,"
    cases = [
        ("greedy", {"do_sample": False}),
        ("sample_temp_0.7", {"do_sample": True, "temperature": 0.7, "top_p": 0.95}),
        ("sample_temp_1.2", {"do_sample": True, "temperature": 1.2, "top_p": 0.95}),
        ("beam_3", {"do_sample": False, "num_beams": 3}),
    ]

    for name, kwargs in cases:
        print(f"\n== {name} ==")
        for i in range(3):
            print(f"[{i + 1}] {generate_case(model, tokenizer, prompt, **kwargs)}")


if __name__ == "__main__":
    main()

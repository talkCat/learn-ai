"""Run several Hugging Face pipeline tasks.

This script is intentionally small: the goal is to see the engineering shape
of tokenizer -> model -> task wrapper before reading model internals.
"""

from transformers import pipeline


def main() -> None:
    sentiment = pipeline("sentiment-analysis")
    qa = pipeline("question-answering")
    generator = pipeline("text-generation", model="sshleifer/tiny-gpt2")

    print("\n== sentiment-analysis ==")
    print(sentiment("Transformers are intimidating at first, but the API helps."))

    print("\n== question-answering ==")
    print(
        qa(
            question="What does the API help with?",
            context="Transformers are intimidating at first, but the API helps.",
        )
    )

    prompt = "A software engineer learning transformers discovers"
    print("\n== text-generation: max_new_tokens comparison ==")
    for max_new_tokens in [8, 20]:
        output = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=generator.tokenizer.eos_token_id,
        )[0]["generated_text"]
        print(f"\nmax_new_tokens={max_new_tokens}\n{output}")


if __name__ == "__main__":
    main()

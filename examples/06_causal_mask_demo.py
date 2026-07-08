"""Show how a causal mask prevents each position from seeing the future."""

import math

import torch


def main() -> None:
    torch.manual_seed(3)
    seq, dim = 5, 4
    q = torch.randn(seq, dim)
    k = torch.randn(seq, dim)

    scores = q @ k.T / math.sqrt(dim)
    causal_mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    masked_scores = scores.masked_fill(~causal_mask, float("-inf"))
    weights = torch.softmax(masked_scores, dim=-1)

    torch.set_printoptions(precision=3, sci_mode=False)
    print("raw scores:")
    print(scores)
    print("\ncausal mask, 1 means visible:")
    print(causal_mask.int())
    print("\nmasked scores:")
    print(masked_scores)
    print("\nattention weights after softmax:")
    print(weights)
    print("\nEach row can only put probability mass on current or previous columns.")


if __name__ == "__main__":
    main()

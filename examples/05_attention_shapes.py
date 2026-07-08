"""A small scaled dot-product attention demo focused on tensor shapes."""

import math

import torch


def scaled_dot_product_attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights, scores


def main() -> None:
    torch.manual_seed(7)

    batch, seq, dim, heads = 2, 4, 8, 2
    head_dim = dim // heads
    x = torch.randn(batch, seq, dim)

    qkv = torch.nn.Linear(dim, dim * 3, bias=False)
    q, k, v = qkv(x).chunk(3, dim=-1)

    def split_heads(t):
        return t.view(batch, seq, heads, head_dim).transpose(1, 2)

    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)
    output, weights, scores = scaled_dot_product_attention(q, k, v)
    merged = output.transpose(1, 2).contiguous().view(batch, seq, dim)

    print("x:", tuple(x.shape))
    print("q/k/v:", tuple(q.shape), tuple(k.shape), tuple(v.shape))
    print("attention scores:", tuple(scores.shape))
    print("attention weights:", tuple(weights.shape))
    print("attention output per head:", tuple(output.shape))
    print("merged output:", tuple(merged.shape))


if __name__ == "__main__":
    main()

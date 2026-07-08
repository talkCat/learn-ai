"""Character-level dataset helpers for Mini GPT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TextDataset:
    train_data: torch.Tensor
    val_data: torch.Tensor
    stoi: dict[str, int]
    itos: dict[int, str]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_text_dataset(path: str | Path, train_ratio: float = 0.9) -> TextDataset:
    text_path = Path(path)
    text = text_path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

    split = max(1, int(train_ratio * len(ids)))
    train_data = ids[:split]
    val_data = ids[split:] if split < len(ids) else ids[:1]
    return TextDataset(train_data=train_data, val_data=val_data, stoi=stoi, itos=itos)


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError("Dataset is too small for the requested block_size.")
    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y

from typing import Iterator

import mlx.core as mx
import mlx.nn as nn

from constants import BLOCK_SIZE, EMBEDDING_DIMENSIONS, NUM_BLOCKS, NUM_HEADS
from transformer import TransformerLayer


class GPT(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, EMBEDDING_DIMENSIONS)
        self.positional_embedding_table = nn.Embedding(BLOCK_SIZE, EMBEDDING_DIMENSIONS)

        self.blocks = nn.Sequential(
            *[TransformerLayer(num_heads=NUM_HEADS) for _ in range(NUM_BLOCKS)]
        )

        self.layer_norm = nn.LayerNorm(EMBEDDING_DIMENSIONS)
        self.language_model_head = nn.Linear(EMBEDDING_DIMENSIONS, vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        time_steps = x.shape[1]

        token_embeddings = self.token_embedding_table(x)
        position_embeddings = self.positional_embedding_table(mx.arange(time_steps))

        x = token_embeddings + position_embeddings

        x = self.blocks(x)
        x = self.layer_norm(x)

        x = self.language_model_head(x)

        return x

    def generate(self, x: mx.array, num_tokens: int) -> Iterator[int]:
        for _ in range(num_tokens):
            x_cropped = x[:, -BLOCK_SIZE:]
            logits = self(x_cropped)[:, -1, :]
            next_token = mx.random.categorical(logits, axis=-1, num_samples=1)
            x = mx.concatenate([x, next_token], axis=1)
            yield next_token.item()

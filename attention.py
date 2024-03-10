import mlx.core as mx
import mlx.nn as nn

from constants import BLOCK_SIZE, DROPOUT, EMBEDDING_DIMENSIONS


class Head(nn.Module):
    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(EMBEDDING_DIMENSIONS, head_size, bias=False)
        self.query = nn.Linear(EMBEDDING_DIMENSIONS, head_size, bias=False)
        self.value = nn.Linear(EMBEDDING_DIMENSIONS, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

        self.triangle_lower = mx.tril(mx.ones([BLOCK_SIZE, BLOCK_SIZE]))

    def __call__(self, x: mx.array) -> mx.array:
        time_steps = x.shape[1]

        keys = self.key(x)
        queries = self.query(x)

        weights = (queries @ keys.transpose(0, 2, 1)) * (queries.shape[-1] ** -0.5)
        mask = self.triangle_lower[:time_steps, :time_steps] == 0
        weights = mx.where(mask, float("-inf"), weights)
        weights = mx.softmax(weights, axis=-1)
        weights = self.dropout(weights)

        values = self.value(x)
        return weights @ values


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.projection = nn.Linear(EMBEDDING_DIMENSIONS, EMBEDDING_DIMENSIONS)
        self.dropout = nn.Dropout(DROPOUT)

    def __call__(self, x: mx.array) -> mx.array:
        out = mx.concatenate([head(x) for head in self.heads], axis=-1)
        return self.dropout(self.projection(out))

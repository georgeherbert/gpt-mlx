import math

import mlx.core as mx
import mlx.nn as nn

from constants import BLOCK_SIZE, DROPOUT, EMBEDDING_DIMENSIONS


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int) -> None:
        super().__init__()

        self.num_heads = num_heads

        self.query_projection = nn.Linear(EMBEDDING_DIMENSIONS, EMBEDDING_DIMENSIONS, bias=False)
        self.key_projection = nn.Linear(EMBEDDING_DIMENSIONS, EMBEDDING_DIMENSIONS, bias=False)
        self.value_projection = nn.Linear(EMBEDDING_DIMENSIONS, EMBEDDING_DIMENSIONS, bias=False)
        self.output_projection = nn.Linear(EMBEDDING_DIMENSIONS, EMBEDDING_DIMENSIONS, bias=False)

        self.attention_dropout = nn.Dropout(DROPOUT)
        self.output_dropout = nn.Dropout(DROPOUT)

        self.triangle_lower = mx.tril(mx.ones([BLOCK_SIZE, BLOCK_SIZE]))

        # TODO: Ensure that the number of heads divides the embedding dimensions?
        self.scale = 1 / math.sqrt(EMBEDDING_DIMENSIONS / num_heads)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, time_steps, _ = x.shape

        # (batch_size, time_steps, num_heads, head_size)
        queries = self.query_projection(x).reshape(batch_size, time_steps, self.num_heads, -1)
        keys = self.key_projection(x).reshape(batch_size, time_steps, self.num_heads, -1)
        values = self.value_projection(x).reshape(batch_size, time_steps, self.num_heads, -1)

        queries = queries.transpose(0, 2, 1, 3)  # (batch_size, num_heads, time_steps, head_size)
        keys = keys.transpose(0, 2, 3, 1)  # (batch_size, num_heads, head_size, time_steps)
        values = values.transpose(0, 2, 1, 3)  # (batch_size, num_heads, time_steps, head_size)

        mask = self.triangle_lower[:time_steps, :time_steps] == 0

        # (batch_size, num_heads, time_steps, time_steps)
        scores = (queries @ keys) * self.scale
        scores = mx.where(mask, float("-inf"), scores)
        scores = mx.softmax(scores, axis=-1)
        scores = self.attention_dropout(scores)

        output = scores @ values  # (batch_size, num_heads, time_steps, head_size)
        output = output.transpose(0, 2, 1, 3)  # (batch_size, time_steps, num_heads, head_size)

        # (batch_size, time_steps, embedding_dimensions)
        output = output.reshape(batch_size, time_steps, -1)
        output = self.output_dropout(self.output_projection(output))

        return output

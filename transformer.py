import mlx.core as mx
import mlx.nn as nn

from attention import MultiHeadAttention
from constants import DROPOUT, EMBEDDING_DIMENSIONS


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fully_connected = nn.Sequential(
            nn.Linear(EMBEDDING_DIMENSIONS, 4 * EMBEDDING_DIMENSIONS),
            nn.ReLU(),
            nn.Linear(4 * EMBEDDING_DIMENSIONS, EMBEDDING_DIMENSIONS),
            nn.Dropout(DROPOUT),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.fully_connected(x)


class TransformerLayer(nn.Module):
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        head_size = EMBEDDING_DIMENSIONS // num_heads

        # TODO: Ensure that the number of heads divides the embedding dimensions?

        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward()
        self.layer_norm_1 = nn.LayerNorm(EMBEDDING_DIMENSIONS)
        self.layer_norm_2 = nn.LayerNorm(EMBEDDING_DIMENSIONS)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

import mlx.core as mx
from mlx.nn.layers.activations import ReLU
from mlx.nn.layers.base import Module
from mlx.nn.layers.containers import Sequential
from mlx.nn.layers.dropout import Dropout
from mlx.nn.layers.linear import Linear
from mlx.nn.layers.normalization import LayerNorm

from attention import MultiHeadAttention
from constants import DROPOUT, EMBEDDING_DIMENSIONS


class FeedForward(Module):
    def __init__(self) -> None:
        super().__init__()
        self.fully_connected = Sequential(
            Linear(EMBEDDING_DIMENSIONS, 4 * EMBEDDING_DIMENSIONS),
            ReLU(),
            Linear(4 * EMBEDDING_DIMENSIONS, EMBEDDING_DIMENSIONS),
            Dropout(DROPOUT),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.fully_connected(x)


class TransformerLayer(Module):
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        head_size = EMBEDDING_DIMENSIONS // num_heads

        # TODO: Ensure that the number of heads divides the embedding dimensions?

        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward()
        self.layer_norm_1 = LayerNorm(EMBEDDING_DIMENSIONS)
        self.layer_norm_2 = LayerNorm(EMBEDDING_DIMENSIONS)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

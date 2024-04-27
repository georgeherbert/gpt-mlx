from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import numpy.typing as npt
from mlx.utils import tree_flatten, tree_map

from constants import ACCUMULATION_STEPS, BATCH_SIZE, BLOCK_SIZE, ITERATIONS, LEARNING_RATE
from gpt import GPT


def get_batch(data: npt.NDArray) -> tuple[mx.array, mx.array]:
    indices = np.random.randint(0, data.shape[0] - BLOCK_SIZE, BATCH_SIZE)
    # indices = mx.random.randint(0, data.shape[0] * 1. - BLOCK_SIZE, [BATCH_SIZE])
    # TODO: Investigate why the above line is not working.

    x = mx.stack([mx.array(data[i.item() : i.item() + BLOCK_SIZE]) for i in indices])
    y = mx.stack([mx.array(data[i.item() + 1 : i.item() + BLOCK_SIZE + 1]) for i in indices])

    return x, y


def loss_function(model: GPT, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    return nn.losses.cross_entropy(logits, y, reduction="mean")


def save_checkpoint(model: GPT, iteration: int) -> None:
    checkpoint_path = Path(f"checkpoints/{iteration}")
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    model.save_weights(str(checkpoint_path / "model.safetensors"))


def main() -> None:
    train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")

    model = GPT(vocab_size=50257)  # GPT2 vocab size
    optimiser = optim.AdamW(learning_rate=LEARNING_RATE)

    num_parameters = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Training a GPT model with {num_parameters / 1e6:.3f} million parameters.")

    loss_and_gradient_function = nn.value_and_grad(model, loss_function)

    state = [model.state, optimiser.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x: mx.array, y: mx.array, accumulated_gradients: dict, i: int) -> tuple[mx.array, dict]:
        loss, gradients = loss_and_gradient_function(model, x, y)
        accumulated_gradients = tree_map(lambda acc, new: acc + new, accumulated_gradients, gradients)

        if i % ACCUMULATION_STEPS == 0:
            optimiser.update(model, accumulated_gradients)
            accumulated_gradients = tree_map(lambda x: mx.zeros_like(x), model.parameters())

        return loss, accumulated_gradients

    accumulated_gradients = tree_map(lambda x: mx.zeros_like(x), model.parameters())

    for i in range(ITERATIONS):
        x, y = get_batch(train_data)

        loss, accumulated_gradients = step(x, y, accumulated_gradients, i)
        mx.eval(state, accumulated_gradients)

        if i % 500 == 0 or i == ITERATIONS - 1:
            save_checkpoint(model, i)

        print(i, loss.item())


if __name__ == "__main__":
    main()

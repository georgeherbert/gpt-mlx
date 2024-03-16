from functools import partial
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from constants import BATCH_SIZE, BLOCK_SIZE, LEARNING_RATE
from gpt import GPT


def get_encoder(characters: list[str]) -> Callable[[str], list[int]]:
    character_to_integer = {character: i for i, character in enumerate(characters)}
    return lambda string: [character_to_integer[char] for char in string]


def get_decoder(characters: list[str]) -> Callable[[list[int]], str]:
    integer_to_character = {i: character for i, character in enumerate(characters)}
    return lambda integers: "".join(integer_to_character[i] for i in integers)


def get_train_test_split(data: mx.array, train_proportion: float = 0.9) -> tuple[mx.array, mx.array]:
    train_size = int(data.shape[0] * train_proportion)
    return data[:train_size], data[train_size:]


def get_batch(data: mx.array) -> tuple[mx.array, mx.array]:
    indices = mx.random.randint(0, data.shape[0] - BLOCK_SIZE, [BATCH_SIZE])

    x = mx.stack([data[i.item() : i.item() + BLOCK_SIZE] for i in indices])
    y = mx.stack([data[i.item() + 1 : i.item() + BLOCK_SIZE + 1] for i in indices])

    return x, y


def loss_function(model: GPT, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    return nn.losses.cross_entropy(logits, y, reduction="mean")


def save_optimiser(optimiser: optim.Optimizer, path: Path) -> None:
    params_dict = dict(tree_flatten(optimiser.state))
    mx.save_safetensors(str(path), params_dict)


def save_checkpoint(model: GPT, optimiser: optim.Optimizer, iteration: int) -> None:
    checkpoint_path = Path(f"checkpoints/{iteration}")
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    model.save_weights(str(checkpoint_path / "model.safetensors"))
    save_optimiser(optimiser, checkpoint_path / "optimiser.safetensors")


# TODO: Find a proper way to load the optimiser.
# def load_optimiser(optimiser: optim.Optimizer, path: Path) -> None:
#     params = list(mx.load(str(path)).items())
#     optimiser.state = tree_unflatten(params)


# def load_checkpoint(model: GPT, optimiser: optim.Optimizer, iteration: int) -> None:
#     checkpoint_path = Path(f"checkpoints/{iteration}")
#     model.load_weights(str(checkpoint_path / "model.safetensors"))
#     load_optimiser(optimiser, checkpoint_path / "optimiser.safetensors")


def main() -> None:
    text = Path("data/tiny_shakespeare.txt").read_text(encoding="utf-8")

    characters = sorted(list(set(text)))
    encoder = get_encoder(characters)
    decoder = get_decoder(characters)

    data = mx.array(encoder(text))
    train_data, test_data = get_train_test_split(data)

    model = GPT(vocab_size=len(characters))
    optimiser = optim.AdamW(learning_rate=LEARNING_RATE)

    loss_and_gradient_function = nn.value_and_grad(model, loss_function)

    state = [model.state, optimiser.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, gradient = loss_and_gradient_function(model, x, y)
        optimiser.update(model, gradient)
        return loss

    iterations = 3000
    for i in range(optimiser.step.item(), iterations):
        x, y = get_batch(train_data)
        loss = step(x, y)
        mx.eval(state)

        if i % 500 == 0 or i == iterations - 1:
            save_checkpoint(model, optimiser, i)

        print(i, loss.item())


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Callable

import mlx.core as mx

from gpt import GPT
from train import get_decoder, get_encoder


def generate(
    model: GPT,
    decoder: Callable[[list[int]], str],
    num_tokens: int,
) -> None:
    context = mx.zeros((1, 1), dtype=mx.uint32)
    for new_token in model.generate(context, num_tokens):
        print(decoder([new_token]), end="", flush=True)


def main() -> None:
    # TODO: Store the tokens in a file to prevent dynamic reinitialisation of the tokeniser.
    text = Path("data/tiny_shakespeare.txt").read_text(encoding="utf-8")
    characters = sorted(list(set(text)))
    decoder = get_decoder(characters)
    global encoder
    encoder = get_encoder(characters)

    model = GPT(vocab_size=len(characters))
    model.load_weights("checkpoints/2500/model.safetensors")

    generate(model, decoder, 5000)


if __name__ == "__main__":
    main()

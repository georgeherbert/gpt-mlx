import mlx.core as mx
import tiktoken

from gpt import GPT


def generate(model: GPT, num_tokens: int) -> None:
    encoder = tiktoken.get_encoding("gpt2")
    context = mx.zeros((1, 1), dtype=mx.uint32)
    for new_token in model.generate(context, num_tokens):
        new_text = encoder.decode([new_token])
        print(new_text, end="", flush=True)


def main() -> None:
    # TODO: Store the tokens in a file to prevent dynamic reinitialisation of the tokeniser.
    model = GPT(vocab_size=50257)  # TODO: Remove magic number
    model.load_weights("checkpoints/2500/model.safetensors")

    generate(model, 5000)


if __name__ == "__main__":
    main()

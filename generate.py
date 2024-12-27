import mlx.core as mx
import tiktoken

from gpt import GPT


def unconditional_context() -> mx.array:
    return mx.zeros((1, 1), dtype=mx.uint32)


def conditional_context(encoder: tiktoken.Encoding, prompt: str) -> mx.array:
    return mx.reshape(mx.array(encoder.encode(prompt), dtype=mx.uint32), (1, -1))


def generate(model: GPT, num_tokens: int, temperature: float = 1.0, prompt: str | None = None) -> None:
    encoder = tiktoken.get_encoding("gpt2")
    context = conditional_context(encoder, prompt) if prompt else unconditional_context()

    for new_token in model.generate(context, num_tokens, temperature):
        new_text = encoder.decode([new_token])
        print(new_text, end="", flush=True)


def main() -> None:
    model = GPT(vocab_size=50257)  # TODO: Magic number
    model.load_weights("checkpoints/20241224_154913/2000/model.safetensors")

    generate(model=model, num_tokens=5000, temperature=0.95)


if __name__ == "__main__":
    main()

import argparse
import math
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import numpy.typing as npt
from mlx.utils import tree_flatten, tree_map

from constants import (
    ACCUMULATION_STEPS,
    BATCH_SIZE,
    BETA1,
    BETA2,
    BLOCK_SIZE,
    CHECKPOINT_INTERVAL,
    ITERATIONS,
    LR_DECAY_ITERS,
    LR_MAX,
    LR_MIN,
    LR_WARMUP_ITERS,
    WEIGHT_DECAY,
)
from gpt import GPT
from utils import get_checkpoint_path, load_checkpoint, save_checkpoint


def get_batch(data: npt.NDArray) -> tuple[mx.array, mx.array]:
    indices = np.random.randint(0, data.shape[0] - BLOCK_SIZE, BATCH_SIZE)
    # indices = mx.random.randint(0, data.shape[0] * 1. - BLOCK_SIZE, [BATCH_SIZE])
    # TODO: Investigate why the above line is not working.

    x = mx.stack([mx.array(data[i.item(): i.item() + BLOCK_SIZE]) for i in indices])
    y = mx.stack([mx.array(data[i.item() + 1: i.item() + BLOCK_SIZE + 1]) for i in indices])

    return x, y


def loss_function(model: GPT, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    return nn.losses.cross_entropy(logits, y, reduction="mean")


def get_learning_rate(step_num: int) -> float:
    # 1) Linear warmup for a warmup period steps
    if step_num < LR_WARMUP_ITERS:
        return LR_MAX * step_num / LR_WARMUP_ITERS
    # 2) If the step num is greater than the decay iterations, we return the minimum learning rate
    if step_num > LR_DECAY_ITERS:
        return LR_MIN
    # 3) Between the above, we use a cosine decay down to the minimum learning rate
    decay_ratio = (step_num - LR_WARMUP_ITERS) / (LR_DECAY_ITERS - LR_WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return LR_MIN + coeff * (LR_MAX - LR_MIN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT Training")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint")
    return parser.parse_args()


def train(args) -> None:
    train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")

    model = GPT(vocab_size=50257)  # GPT2 vocab size
    optimizer = optim.AdamW(
        learning_rate=LR_MAX,
        betas=[BETA1, BETA2],
        weight_decay=WEIGHT_DECAY,
    )

    if args.checkpoint:
        load_checkpoint(model, optimizer, args.checkpoint)

    checkpoints_path = get_checkpoint_path()

    num_parameters = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Training a GPT model with {num_parameters / 1e6:.3f} million parameters.")

    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step_acc(batch_x: mx.array, batch_y: mx.array, prev_acc_gradients: dict) -> tuple[mx.array, dict]:
        batch_loss, batch_gradients = nn.value_and_grad(model, loss_function)(model, batch_x, batch_y)
        new_acc_gradients = tree_map(lambda a, b: a + b, prev_acc_gradients, batch_gradients)
        return batch_loss, new_acc_gradients

    @partial(mx.compile, inputs=state, outputs=state)
    def step_apply(fully_acc_gradients: dict) -> None:
        normalised_acc_gradients = tree_map(lambda a: a / ACCUMULATION_STEPS, fully_acc_gradients)
        optimizer.update(model, normalised_acc_gradients)

    train_loss_path = checkpoints_path / "train_loss.csv"

    for step_num in range(optimizer.step.item(), ITERATIONS):
        total_loss = 0.0
        acc_gradients = tree_map(lambda x: mx.zeros_like(x), model.parameters())

        optimizer.learning_rate = get_learning_rate(step_num)

        for micro_step_num in range(ACCUMULATION_STEPS):
            batch = get_batch(train_data)
            loss, acc_gradients = step_acc(*batch, acc_gradients)

            if micro_step_num == ACCUMULATION_STEPS - 1:
                step_apply(acc_gradients)

            mx.eval(loss, state, acc_gradients)
            total_loss += loss.item()

            print(
                f"Step [{step_num}]\t\t"
                f"Micro Step [{micro_step_num}]\t\t"
                f"Training Loss [{loss.item():.4f}]\t\t"
                f"Learning Rate [{optimizer.learning_rate.item()}]"
            )

        if step_num > 0 and (step_num % CHECKPOINT_INTERVAL == 0 or step_num == ITERATIONS - 1):
            save_checkpoint(model, optimizer, checkpoints_path / str(step_num))

        mean_loss = total_loss / ACCUMULATION_STEPS
        with train_loss_path.open("a+") as f:
            f.write(f"{step_num},{mean_loss}\n")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()

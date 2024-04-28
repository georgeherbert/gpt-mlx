from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from gpt import GPT


def get_checkpoint_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = Path(f"checkpoints") / timestamp
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    return checkpoint_path


def save_checkpoint(model: GPT, optimizer: optim.Optimizer, checkpoint_path: Path) -> None:
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    model.save_weights((checkpoint_path / "model.safetensors").as_posix())

    optimizer_state_dict = dict(tree_flatten(optimizer.state))
    mx.save_safetensors(
        file=(checkpoint_path / "optimizer.safetensors").as_posix(),
        arrays=optimizer_state_dict,
    )


def load_checkpoint(model: GPT, optimizer: optim.Optimizer, checkpoint_path: Path) -> None:
    model.load_weights((checkpoint_path / "model.safetensors").as_posix())
    optimizer_state_dict = mx.load((checkpoint_path / "optimizer.safetensors").as_posix())
    optimizer.state = tree_unflatten(list(optimizer_state_dict.items()))

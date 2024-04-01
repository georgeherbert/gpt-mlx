from pathlib import Path
from typing import Callable

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

from constants import ENCODING

BATCHES = 1024
NUM_PROCESSES = 8


def get_encode() -> Callable[[str], list[int]]:
    encoding = tiktoken.get_encoding(ENCODING)

    def encode(example: str) -> list[int]:
        tokens = encoding.encode_ordinary(example) + [encoding.eot_token]
        return {"tokens": tokens, "length": len(tokens)}

    return encode


def main() -> None:
    dataset = load_dataset("openwebtext", num_proc=NUM_PROCESSES, trust_remote_code=True)

    split_dataset = dataset["train"].train_test_split(test_size=0.0005, shuffle=True)
    split_dataset["validation"] = split_dataset.pop("test")

    split_dataset = split_dataset.map(
        function=get_encode(),
        input_columns=["text"],
        remove_columns=["text"],
        num_proc=NUM_PROCESSES,
    )

    Path("data").mkdir(exist_ok=True)

    for split, data in split_dataset.items():
        file_name = f"data/{split}.bin"
        concatenated = np.memmap(file_name, dtype=np.uint16, mode="w+", shape=(sum(data["length"]),))

        concatenated_index = 0
        for batch_index in tqdm(range(BATCHES), desc=f"Writing {file_name}"):
            batch = data.shard(num_shards=BATCHES, index=batch_index, contiguous=True).with_format("numpy")
            batch_tokens = np.concatenate(batch["tokens"])
            batch_tokens_size = batch_tokens.shape[0]

            concatenated[concatenated_index : concatenated_index + batch_tokens_size] = batch_tokens
            concatenated_index += batch_tokens_size

        concatenated.flush()


if __name__ == "__main__":
    main()

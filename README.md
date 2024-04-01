# GPT MLX

## Introduction

This is a simple implementation of a GPT using Apple's new [MLX](https://ml-explore.github.io/mlx/build/html/index.html) library.

## Installation

Ensure [Poetry](https://python-poetry.org/docs/) is installed and run:

```bash
poetry install
```

## Usage

### Training

We train the model on [OpenWebText](https://huggingface.co/datasets/openwebtext), which is an open-source replication of OpenAI's WebText dataset.

Run the following command to download the dataset, tokenize it and then save it to disk:

```bash
poetry run python prepare.py
```

This will create a `data` directory with two files: `train.bin` and `validation.bin`.

Once complete, run the following command to train the model:

```bash
poetry run python train.py
```

Checkpoints will be saved to the `checkpoints` directory.

### Generation

Run the following command to generate text using the trained model:

```bash
poetry run python generate.py
```

## Acknowledgments

This implementation has been inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [minGPT](https://github.com/karpathy/minGPT) repositories, which are themselves PyTorch reimplementations of [GPT-2](https://github.com/openai/gpt-2) with a few modifications.

## TODO

- Configuration improvements (e.g. YAML file)
- Calculate validation loss
- Adjust hyperparameters to improve performance

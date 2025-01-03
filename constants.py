ITERATIONS = 100000

# 128 * 2 * 1024 = 262144 tokens per optimisation step
ACCUMULATION_STEPS = 128
BATCH_SIZE = 2
BLOCK_SIZE = 1024

LR_MAX = 6e-5
LR_MIN = 6e-6
LR_WARMUP_ITERS = 1000
LR_DECAY_ITERS = 100000
WEIGHT_DECAY = 1
BETA1 = 0.95
BETA2 = 0.98

EMBEDDING_DIMENSIONS = 768
NUM_BLOCKS = 12
NUM_HEADS = 12

DROPOUT = 0.0

ENCODING = "gpt2"

CHECKPOINT_INTERVAL = 100

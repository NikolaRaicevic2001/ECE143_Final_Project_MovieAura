import torch

SEQ_LEN = 20            # Number of past movies to consider
EMBEDDING_DIM = 128     # Dimension of the movie embedding
BATCH_SIZE = 32         # Number of samples per batch
NUM_LAYERS = 2          # Number of transformer layers
NUM_HEADS = 4           # Number of attention heads
DROPOUT_RATE = 0.2      # Dropout rate for the transformer
EPOCHS = 30             # Number of training epochs
INPUT_DIM = 1707        # Dimension of the input features
NUM_MOVIES = 14010      # Number of movies in the dataset
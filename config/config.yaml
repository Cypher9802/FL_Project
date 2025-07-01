import torch
import os

def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class Config:
    # Dataset
    NUM_CLASSES = 6
    INPUT_SIZE = 128
    INPUT_FEATURES = 9

    # Federated Learning
    TOTAL_CLIENTS = 30
    CLIENT_FRACTION = 10 / 30  # 10 clients per round
    ROUNDS = 30
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 16

    # Privacy
    EPSILON = 8.0
    DELTA = 1e-5
    MAX_GRAD_NORM = 1.0
    NOISE_MULTIPLIER = 1.2

    # Model
    TARGET_ACCURACY = 0.95
    MAX_MODEL_SIZE_MB = 5
    MODEL_SAVE_PATH = "models/fl_model.pt"

    # Training
    DEVICE = resolve_device()
    LR = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4

    # Security
    USE_SECURE_AGGREGATION = True
    AGGREGATION_NOISE_SCALE = 0.1

    @staticmethod
    def create_dirs():
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)

Config.create_dirs()

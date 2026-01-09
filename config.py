"""
Configuration file for hyperparameter tuning.
Modify values here to experiment with different settings.
"""

# Dataset configuration
DATASETS = {
    "rice": "data/Wfp_rice.csv",
    "wheat": "data/Wfp_wheat.csv",
}

# Data split
VAL_SIZE = 24
TEST_SIZE = 24

# Window configuration
LOOKBACK = 60  # Historical context window
HORIZON = 6    # Forecast horizon
PATCH_LEN = 4  # Patch length for PatchTST

# Training configuration
TEACHER_EPOCHS = 300
STUDENT_EPOCHS = 400
SEED = 42

# Knowledge distillation
ALPHA0 = 0.6  # Initial KD weight (decreases linearly to 0)

# Optimizer settings
LEARNING_RATE = 5e-4    # Initial learning rate
WEIGHT_DECAY = 1e-4     # L2 regularization
PATIENCE = 50           # Early stopping patience

# Student architecture
HIDDEN_DIM = 256   # Hidden dimension size
N_BLOCKS = 3       # Number of residual blocks
DROPOUT = 0.15     # Dropout rate

# Data augmentation
AUGMENT_NOISE_STD = 0.015        # Gaussian noise std
AUGMENT_SCALE_RANGE = (0.97, 1.03)  # Random scaling range

# Advanced settings
GRADIENT_CLIP_NORM = 1.0  # Max gradient norm
LR_MIN_FACTOR = 0.01      # Min LR = LEARNING_RATE * LR_MIN_FACTOR


# Preset configurations for different scenarios
PRESETS = {
    "default": {
        "LOOKBACK": 60,
        "HIDDEN_DIM": 256,
        "N_BLOCKS": 3,
        "LEARNING_RATE": 5e-4,
        "ALPHA0": 0.6,
    },
    "high_capacity": {
        "LOOKBACK": 72,
        "HIDDEN_DIM": 384,
        "N_BLOCKS": 4,
        "LEARNING_RATE": 3e-4,
        "ALPHA0": 0.65,
    },
    "fast": {
        "LOOKBACK": 48,
        "HIDDEN_DIM": 128,
        "N_BLOCKS": 2,
        "LEARNING_RATE": 1e-3,
        "ALPHA0": 0.5,
        "TEACHER_EPOCHS": 150,
        "STUDENT_EPOCHS": 200,
    },
    "robust": {
        "LOOKBACK": 60,
        "HIDDEN_DIM": 256,
        "N_BLOCKS": 3,
        "LEARNING_RATE": 3e-4,
        "ALPHA0": 0.7,
        "DROPOUT": 0.2,
        "AUGMENT_NOISE_STD": 0.02,
    },
}


def apply_preset(preset_name):
    """Apply a preset configuration."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")

    preset = PRESETS[preset_name]
    globals().update(preset)
    print(f"Applied preset: {preset_name}")


def print_config():
    """Print current configuration."""
    print("\n" + "=" * 60)
    print("CURRENT CONFIGURATION")
    print("=" * 60)
    print(f"Lookback:         {LOOKBACK}")
    print(f"Horizon:          {HORIZON}")
    print(f"Teacher Epochs:   {TEACHER_EPOCHS}")
    print(f"Student Epochs:   {STUDENT_EPOCHS}")
    print(f"Learning Rate:    {LEARNING_RATE}")
    print(f"Alpha0 (KD):      {ALPHA0}")
    print(f"Hidden Dim:       {HIDDEN_DIM}")
    print(f"Residual Blocks:  {N_BLOCKS}")
    print(f"Dropout:          {DROPOUT}")
    print(f"Augment Noise:    {AUGMENT_NOISE_STD}")
    print("=" * 60 + "\n")

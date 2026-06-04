"""Project paths and hyperparameters."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "COVID_19_dataset"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SALIENCY_DIR = OUTPUTS_DIR / "saliency_maps"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"

CLASS_NAMES = ["COVID", "Normal", "Viral Pneumonia"]
NUM_CLASSES = 3

IMAGE_SIZE = 256
BATCH_SIZE = 16
NUM_WORKERS = 0  # Windows: avoid multiprocessing dataloader hangs

LR = 3e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 5
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5

BEST_MODEL_PATH = MODELS_DIR / "densenet121_best.pt"
CHECKPOINT_PATH = MODELS_DIR / "densenet121_last.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

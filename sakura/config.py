"""Project-level paths and constants."""

from pathlib import Path

# Absolute path to the project root (parent of this package)
PROJECT_ROOT = Path(__file__).parent.parent

# All downloaded model files live here (gitignored)
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

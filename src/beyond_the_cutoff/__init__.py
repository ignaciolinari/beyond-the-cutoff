"""Beyond the Cutoff research assistant toolkit."""

from .config import ProjectConfig, load_config
from .models import OllamaClient

__all__ = ["ProjectConfig", "load_config", "OllamaClient", "__version__"]

__version__ = "0.1.0"

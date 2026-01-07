# src/karhu/models/__init__.py
from .gmax import GMaxPredictor, load_model
# from .gntor import GNtorPredictor

__all__ = [
    "GMaxPredictor", "load_model"
    # "GNtorPredictor",
]

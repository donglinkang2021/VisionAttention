from dataclasses import dataclass
from omegaconf import MISSING

@dataclass
class DatasetConfig:
    name: str = MISSING
    num_classes: int = MISSING
from dataclasses import dataclass
from omegaconf import MISSING

@dataclass
class TrainConfig:
    batch_size: int = MISSING
    num_epochs: int = MISSING
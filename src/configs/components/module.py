from dataclasses import dataclass
from omegaconf import MISSING

@dataclass
class ModuleConfig:
    backbone: str = MISSING
    learning_rate: float = MISSING
    compile: bool = MISSING
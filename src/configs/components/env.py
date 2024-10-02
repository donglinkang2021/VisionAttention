from dataclasses import dataclass
from omegaconf import MISSING

@dataclass
class EnvConfig:
    data_root: str = MISSING
    torch_home: str = MISSING
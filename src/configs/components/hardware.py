from dataclasses import dataclass
from omegaconf import MISSING
from typing import List

@dataclass
class HardwareConfig:
    num_workers: int = MISSING
    accelerator: str = MISSING
    devices: List[int] = MISSING
    precision: int = MISSING
    num_nodes: int = MISSING
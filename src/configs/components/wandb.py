from dataclasses import dataclass
from omegaconf import MISSING
from typing import Union

@dataclass
class WandbConfig:
    project: str = MISSING
    offline: bool = MISSING
    log_model: Union[bool, str] = MISSING
    
from dataclasses import dataclass
from omegaconf import MISSING

from .components.env import EnvConfig
from .components.module import ModuleConfig
from .components.dataset import DatasetConfig
from .components.hardware import HardwareConfig
from .components.train import TrainConfig
from .components.wandb import WandbConfig

@dataclass
class Config:
    env: EnvConfig = MISSING
    module: ModuleConfig = MISSING
    dataset: DatasetConfig = MISSING
    hardware: HardwareConfig = MISSING
    train: TrainConfig = MISSING
    wandb: WandbConfig = MISSING
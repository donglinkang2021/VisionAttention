import os
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import hydra
import wandb
from omegaconf import DictConfig
from src.configs.config import Config
from src.tasks.image_classification import ImageClassificationModule
from src.datasets import get_datamodule
from src.models import get_model

torch.set_float32_matmul_precision('high')

def set_env(cfg: Config) -> None:
    from pathlib import Path
    DATA_ROOT = cfg.env.data_root
    TORCHHOME = cfg.env.torch_home
    Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    Path(TORCHHOME).mkdir(parents=True, exist_ok=True)
    os.environ['TORCH_HOME'] = TORCHHOME
    os.environ['HYDRA_FULL_ERROR'] = '1'

@hydra.main(config_path="conf", config_name="default", version_base=None)
def my_app(cfg: Config) -> None:
    # print(OmegaConf.to_yaml(cfg))
    # return
    set_env(cfg)

    dm = get_datamodule(
        dataset_name = cfg.dataset.name,
        data_dir = cfg.env.data_root, 
        batch_size = cfg.train.batch_size,
        num_workers = cfg.hardware.num_workers
    )

    model = get_model(cfg.module.backbone, cfg.dataset.num_classes)
    optim_partial = hydra.utils.instantiate(cfg.optimizer, lr=cfg.module.learning_rate)
    scheduler_partial = hydra.utils.instantiate(cfg.scheduler)
    if isinstance(scheduler_partial, DictConfig):
        scheduler_partial = None
    
    lm = ImageClassificationModule(
        num_classes = cfg.dataset.num_classes,
        model = model,
        optimizer = optim_partial,
        scheduler = scheduler_partial,
        compile = cfg.module.compile
    )

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        offline=cfg.wandb.offline,
        log_model=cfg.wandb.log_model
    )

    trainer = L.Trainer(
        logger=wandb_logger,
        accelerator = cfg.hardware.accelerator,
        devices = cfg.hardware.devices,
        precision = cfg.hardware.precision,
        num_nodes = cfg.hardware.num_nodes,
        max_epochs = cfg.train.num_epochs
    )

    trainer.fit(lm, dm)
    trainer.validate(lm, dm)
    trainer.test(lm, dm)
    wandb.finish()


if __name__ == "__main__":
    my_app()
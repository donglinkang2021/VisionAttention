from pathlib import Path
from datetime import datetime

outdir = Path("/root/autodl-tmp/output")
outdir.mkdir(parents=True, exist_ok=True)

def save_ckpt(model_name:str) -> Path:
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_ckpts = outdir / "checkpoints" / model_name / current_time 
    model_ckpts.mkdir(parents=True, exist_ok=True)
    return model_ckpts

if __name__ == "__main__":
    print(save_ckpt("resnet18"))
import torch.nn as nn

def get_num_params(model:nn.Module):
    """
    Usage
    ---
    >>> print(f"number of parameters: {get_num_params(model)/1e6:.6f} M ")
    """
    return sum(p.numel() for p in model.parameters())

def show_model_size(model:nn.Module):
    """
    Usage
    ---
    >>> show_model_size(model)
    """
    number_of_params = get_num_params(model)
    print(f"number of parameters: {number_of_params/1e6:.6f} M ")
    model_size = number_of_params*4 / 2**20
    print(f"model size: {model_size:.6f} MB")

if __name__ == "__main__":
    import os
    from pathlib import Path
    outdir = "/root/autodl-tmp/.cache/torch"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    os.environ['TORCH_HOME'] = outdir

    import torchvision.models as models
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    show_model_size(model)
    print(model, file=open(".cache/resnet18.txt", "w"))
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    show_model_size(model)
    print(model, file=open(".cache/resnet34.txt", "w"))
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    show_model_size(model)
    print(model, file=open(".cache/resnet50.txt", "w"))
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    show_model_size(model)
    print(model, file=open(".cache/resnet101.txt", "w"))
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    show_model_size(model)
    print(model, file=open(".cache/resnet152.txt", "w"))

# python -m src.utils.show
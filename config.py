from models import save_ckpt

## model config
is_pretrained = False
in_channels = 3
n_channels = 64
n_classes = 10
n_head = None

## train config
batch_size = 512
learning_rate = 1e-3
num_epochs = 10
eval_interval = 100
save_begin = 500

## save config
# model_name = 'CNN'
pretrained_backbone = 'resnet18'
classifier_name = 'linear' if n_head is None else f'heads{n_head}'
model_name = f'{pretrained_backbone}_{classifier_name}'
model_ckpts = save_ckpt(model_name)
print(f"the model checkpoints will be saved at {model_ckpts}.")
## model config
is_pretrained = False
in_channels = 3
n_channels = 64
n_classes = 10
n_head = None
is_freeze = False

## train config
batch_size = 512
learning_rate = 1e-3
num_epochs = 10
eval_interval = 100
save_begin = 500

## save config
pretrained_backbone = 'resnet50'
classifier_name = 'linear' if n_head is None else f'heads{n_head}'
model_name = f'{pretrained_backbone}_{classifier_name}'
python main.py --multirun \
    module.learning_rate=3e-4,3e-3,3e-2 \
    optimizer=adam,sgd \
    scheduler=default,steplr \
    module.backbone=resnet18
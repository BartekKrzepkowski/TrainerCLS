import torch

from src.data.datasets import get_mnist, get_kmnist, get_fmnist, get_svhn, get_cifar10, get_cifar100, get_tinyimagenet
from src.modules.architectures.from_pytorch import get_efficientnet_b0, get_efficientnet_v2_s, get_convnext_t, get_resnet
from src.modules.losses import ClassificationLoss, FisherPenaltyLoss, MSESoftmaxLoss
from src.visualization.clearml_logger import ClearMLLogger
from src.visualization.tensorboard_pytorch import TensorboardPyTorch
from src.visualization.wandb_logger import WandbLogger


ACT_NAME_MAP = {
    'gelu': torch.nn.GELU,
    'identity': torch.nn.Identity,
    'relu': torch.nn.ReLU,
    'sigmoid': torch.nn.Sigmoid,
    'tanh': torch.nn.Tanh,
}

DATASET_NAME_MAP = {
    'mnist': get_mnist,
    'kmnist': get_kmnist,
    'fmnist': get_fmnist,
    'svhn': get_svhn,
    'cifar10': get_cifar10,
    'cifar100': get_cifar100,
    'tinyimagenet': get_tinyimagenet,
}

LOGGERS_NAME_MAP = {
    'clearml': ClearMLLogger,
    'tensorboard': TensorboardPyTorch,
    'wandb': WandbLogger
}

LOSS_NAME_MAP = {
    'ce': torch.nn.CrossEntropyLoss,
    'cls': ClassificationLoss,
    'fp': FisherPenaltyLoss,
    'nll': torch.nn.NLLLoss,
    'mse': torch.nn.MSELoss,
    'mse_softmax': MSESoftmaxLoss,
}

MODEL_NAME_MAP = {
    'convnext': get_convnext_t,
    'effnetv1b0': get_efficientnet_b0,
    'effnetv2s': get_efficientnet_v2_s,
    'resnet': get_resnet,
}

NORM_LAYER_NAME_MAP = {
    'bn1d': torch.nn.BatchNorm1d,
    'bn2d': torch.nn.BatchNorm2d,
    'group_norm': torch.nn.GroupNorm,
    'instance_norm_1d': torch.nn.InstanceNorm1d,
    'instance_norm_2d': torch.nn.InstanceNorm2d,
    'layer_norm': torch.nn.LayerNorm,
}

OPTIMIZER_NAME_MAP = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD,
}

SCHEDULER_NAME_MAP = {
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'cosine_warm_restarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'multiplicative': torch.optim.lr_scheduler.MultiplicativeLR,
    'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
}

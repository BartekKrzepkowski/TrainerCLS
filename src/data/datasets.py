import os

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from src.data.transforms import TRANSFORMS_TRAIN, TRANSFORMS_TEST


DOWNLOAD = False


def get_mnist(dataset_path=None):
    dataset_path = dataset_path if dataset_path is not None else os.environ['MNIST_PATH']
    
    train_dataset = datasets.MNIST(dataset_path, train=True, download=DOWNLOAD, transforms=TRANSFORMS_TRAIN['mnist'])
    train_without_aug_dataset = datasets.MNIST(dataset_path, train=True, download=DOWNLOAD, transforms=TRANSFORMS_TEST['mnist'])
    test_dataset = datasets.MNIST(dataset_path, train=False, download=DOWNLOAD, transforms=TRANSFORMS_TEST['mnist'])
    
    return train_dataset, train_without_aug_dataset, test_dataset


def get_kmnist(dataset_path=None):
    dataset_path = dataset_path if dataset_path is not None else os.environ['KMNIST_PATH']
    
    train_dataset = datasets.KMNIST(dataset_path, train=True, download=DOWNLOAD, transforms=TRANSFORMS_TRAIN['kmnist'])
    train_without_aug_dataset = datasets.KMNIST(dataset_path, train=True, download=DOWNLOAD, transforms=TRANSFORMS_TEST['kmnist'])
    test_dataset = datasets.KMNIST(dataset_path, train=False, download=DOWNLOAD, transforms=TRANSFORMS_TEST['kmnist'])
    
    return train_dataset, train_without_aug_dataset, test_dataset


def get_fmnist(dataset_path=None):
    dataset_path = dataset_path if dataset_path is not None else os.environ['FMNIST_PATH']
    
    train_dataset = datasets.FashionMNIST(dataset_path, train=True, download=DOWNLOAD, transforms=TRANSFORMS_TRAIN['fmnist'])
    train_without_aug_dataset = datasets.FashionMNIST(dataset_path, train=True, download=DOWNLOAD, transforms=TRANSFORMS_TEST['fmnist'])    
    test_dataset = datasets.FashionMNIST(dataset_path, train=False, download=DOWNLOAD, transforms=TRANSFORMS_TEST['fmnist'])
    
    return train_dataset, train_without_aug_dataset, test_dataset


def get_svhn(dataset_path=None):
    dataset_path = dataset_path if dataset_path is not None else os.environ['SVHN_PATH']
    
    train_dataset = datasets.SVHN(dataset_path, split='train', download=DOWNLOAD, transforms=TRANSFORMS_TRAIN['svhn'])    
    train_without_aug_dataset = datasets.SVHN(dataset_path, split='train', download=DOWNLOAD, transforms=TRANSFORMS_TEST['svhn'])    
    test_dataset = datasets.SVHN(dataset_path, split='test', download=DOWNLOAD, transforms=TRANSFORMS_TEST['svhn'])
        
    return train_dataset, train_without_aug_dataset, test_dataset


def get_cifar10(dataset_path=None):
    dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR10_PATH']
    
    train_dataset = datasets.CIFAR10(dataset_path, train=True, download=True, transforms=TRANSFORMS_TRAIN['cifar10'])    
    train_without_aug_dataset = datasets.CIFAR10(dataset_path, train=True, download=True, transforms=TRANSFORMS_TEST['cifar10'])    
    test_dataset = datasets.CIFAR10(dataset_path, train=False, download=True, transforms=TRANSFORMS_TEST['cifar10'])
    
    return train_dataset, train_without_aug_dataset, test_dataset


def get_cifar100(dataset_path=None):
    dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR100_PATH']
    
    train_dataset = datasets.CIFAR100(dataset_path, train=True, download=True, transforms=TRANSFORMS_TRAIN['cifar100'])    
    train_without_aug_dataset = datasets.CIFAR100(dataset_path, train=True, download=True, transforms=TRANSFORMS_TEST['cifar100'])    
    test_dataset = datasets.CIFAR100(dataset_path, train=False, download=True, transforms=TRANSFORMS_TEST['cifar100'])
    
    return train_dataset, train_without_aug_dataset, test_dataset
    

def get_tinyimagenet(dataset_path=True):
    dataset_path = dataset_path if dataset_path is not None else os.environ['TINYIMAGENET_PATH']
    train_path = f'{dataset_path}/train'
    test_path = f'{dataset_path}/val'
   
    train_dataset = datasets.ImageFolder(train_path, transforms=TRANSFORMS_TRAIN['tinyimagenet'])
    train_without_aug_dataset = datasets.ImageFolder(train_path, transforms=TRANSFORMS_TEST['tinyimagenet'])
    test_dataset = datasets.ImageFolder(test_path, transforms=TRANSFORMS_TEST['tinyimagenet'])
    
    return train_dataset, train_without_aug_dataset, test_dataset


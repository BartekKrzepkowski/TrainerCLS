from torchvision import transforms


# train
transform_train_mnist = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_train_fmnist = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

transform_train_svhn = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_train_cifar10 =  transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=5.0, translate=(1/8, 1/8)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))
])

transform_train_cifar100 = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.267, 0.256, 0.276)),
        transforms.RandomErasing(p=0.1)
])

transform_train_tinyimagenet = transforms.Compose([
            transforms.RandomResizedCrop(64, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
            transforms.RandomErasing(p=0.1),
])


# test
transform_eval_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_eval_fmnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

transform_eval_svhn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_eval_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)),
])

transform_eval_cifar100 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.267, 0.256, 0.276)),
])

transform_eval_tinyimagenet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
])


TRANSFORMS_TRAIN = {
    'mnist': transform_train_mnist,
    'fmnist': transform_train_fmnist,
    'svhn': transform_train_svhn,
    'cifar10': transform_train_cifar10,
    'cifar100': transform_train_cifar100,
    'tinyimagenet': transform_train_tinyimagenet,
}

TRANSFORMS_TEST = {
    'mnist': transform_eval_mnist,
    'fmnist': transform_eval_fmnist,
    'svhn': transform_eval_svhn,
    'cifar10': transform_eval_cifar10,
    'cifar100': transform_eval_cifar100,
    'tinyimagenet': transform_eval_tinyimagenet,
}
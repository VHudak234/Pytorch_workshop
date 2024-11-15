from torchvision.datasets import CIFAR10
DATA_ROOT = '../cifar10'

train_dataset = CIFAR10(
    root=DATA_ROOT, train=True, download=True)

test_dataset = CIFAR10(
    root=DATA_ROOT, train=False, download=True)
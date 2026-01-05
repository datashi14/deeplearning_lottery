import torch
from torchvision import datasets, transforms
import os

def get_mnist_loaders(batch_size=64, test_batch_size=1000, data_dir='./data'):
    """
    Returns train_loader, val_loader for MNIST.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    os.makedirs(data_dir, exist_ok=True)
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return train_loader, val_loader

def get_cifar10_loaders(batch_size=128, test_batch_size=100, data_dir='./data'):
    """
    Returns train_loader, val_loader for CIFAR-10.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    os.makedirs(data_dir, exist_ok=True)
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader

def get_loaders(config):
    dataset_name = config['training'].get('dataset', 'mnist').lower()
    batch_size = config['training'].get('batch_size', 64)
    data_dir = config['training'].get('data_dir', './data')
    
    if dataset_name == 'mnist':
        return get_mnist_loaders(batch_size=batch_size, data_dir=data_dir)
    elif dataset_name == 'cifar10':
        return get_cifar10_loaders(batch_size=batch_size, data_dir=data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

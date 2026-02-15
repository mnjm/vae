from torchvision import datasets
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from pathlib import Path
import torch

supported_datasets = ['mnist', 'cifar10']
def init_dataloaders(cfg: object, split: str):
    dataset_name = cfg.dataset_name
    assert dataset_name in supported_datasets, f"Only {supported_datasets} datasets are supported"
    assert split in ['train', 'val']
    cache_dir = Path("./dataset")
    cache_dir.mkdir(exist_ok=True, parents=True)

    img_size = {
        'mnist': (28, 28),
        'cifar10': (32, 32),
    }[dataset_name]
    transform = T.Compose([
        T.Resize(img_size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    train = split == 'train'
    if dataset_name == "mnist":
        dataset = datasets.MNIST(root=cache_dir, train=train, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root=cache_dir, train=train, download=True, transform=transform)

    dataloader_kwargs = dict(cfg.dataloader)
    dataloader_kwargs['drop_last'] = split=="train" and dataloader_kwargs.get('drop_last', False)
    dataloader_kwargs['shuffle'] = split=="train"
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    return dataloader
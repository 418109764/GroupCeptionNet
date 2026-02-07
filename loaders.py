import random

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from scipy.ndimage import zoom
from torch.utils.data.sampler import WeightedRandomSampler

from utils import img_nom
from utils.enums import Tasks
from utils.customdatasets import CustomImageDataset  # 替换 ChiliseedsDataset 为 CustomImageDataset

def _sampler(dataset):
    """Creates a sampler for the dataset."""
    targets = np.array([x[1] for x in dataset.samples])
    total = len(targets)

    samples_weight = np.zeros(total)

    for t in np.unique(targets):
        idx = np.where(targets == t)[0]

        samples_weight[idx] = 1 / (len(idx) / total)

    samples_weight = samples_weight / sum(samples_weight)
    samples_weight = torch.from_numpy(samples_weight).double()

    return WeightedRandomSampler(samples_weight, len(samples_weight))


def _get_transforms(model=None):
    if model == "inception_v3":
        new_size = (299, 299)
    else:
        new_size = (224, 224)

    train_transforms = transforms.Compose(
        [
            transforms.Resize(new_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomApply([transforms.RandomRotation(10)], 0.25),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(new_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_transforms, val_transforms


def _build_loaders(
        train_dataset, val_dataset, test_dataset, batch_size, num_workers=4, pin_memory=True
):
    """Build the loaders."""
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


def custom_image_loader(images_dir, batch_size, csv_file, fold, model_task, num_workers=4,
                        pca_factor=1, target=0, model=None):  # 替换 chiliseeds_loader 为 custom_image_loader
    """Build the dataloaders."""
    train_transforms, val_transforms = _get_transforms(model=model)

    train_dataset = CustomImageDataset(  # 替换 ChiliseedsDataset 为 CustomImageDataset
        csv_file=csv_file,
        images_dir=images_dir,
        dataset="train",
        fold=fold,
        model_task=model_task,
        transforms=train_transforms,
        pca_factor=pca_factor,
        target=target,
    )

    val_dataset = CustomImageDataset(  # 替换 ChiliseedsDataset 为 CustomImageDataset
        csv_file=csv_file,
        images_dir=images_dir,
        dataset="val",
        fold=fold,
        model_task=model_task,
        transforms=val_transforms,
        pca_factor=pca_factor,
        target=target,
    )

    test_dataset = CustomImageDataset(  # 替换 ChiliseedsDataset 为 CustomImageDataset
        csv_file=csv_file,
        images_dir=images_dir,
        dataset="test",
        fold=fold,
        model_task=model_task,
        transforms=val_transforms,
        pca_factor=pca_factor,
        target=target,
    )

    return _build_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        num_workers,
    )
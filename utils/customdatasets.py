# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 10:27:40 2019

@author: Guilherme
"""

import os
import time

import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.enums import Tasks


class ChiliSeedsDataset(Dataset):
    """Coffee Leaves Dataset."""

    def __init__(self, csv_file, images_dir, dataset, fold=1, model_task=None, transforms=None, pca_factor=1, target=1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            images_dir (string): Directory with all the images.
            dataset (string) : Select the desired dataset - 'train', 'val' or 'test'
            fold (int{1,5}) : The data is changed based on the selected fold
            model_task (Tasks) : Select the model task according to the dataset.
            transforms : Image transformations
        """
        self.fold = fold
        self.data = self.split_dataset(pd.read_csv(csv_file), dataset).dropna()
        self.images_dir = images_dir
        self.task = model_task
        self.transformations = transforms
        self.pca_factor = pca_factor
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image name from the pandas df
        img_name = os.path.join(self.images_dir, f"{str(self.data.iloc[idx, 0]).zfill(5)}.jpg")
        image = Image.open(img_name)

        # Apply transformations
        if self.transformations:
            image = self.transformations(image)

        # Get label of the image
        label_germ = self.data.iloc[idx, 1]
        label_germ = torch.tensor(label_germ, dtype=torch.long)

        return image, label_germ

    def split_dataset(self, csv, dataset):
        seed = 122
        np.random.seed(seed)

        dataset_size = len(csv)
        partition_size = int(dataset_size / 5)
        indices = list(range(dataset_size))

        # Get natural samples
        subset_csv = csv[:2524]

        # Filter out natural negative samples and natural positive samples
        negative_indices = subset_csv[subset_csv.iloc[:, 1] == 0]
        positive_indices = subset_csv[subset_csv.iloc[:, 1] == 1]

        # Select 20% of the natural samples from the total samples as the test set.
        n_test_neg = int(0.1 * dataset_size)
        test_neg_indices = np.random.choice(negative_indices.index, n_test_neg, replace=False)
        test_pos_indices = np.random.choice(positive_indices.index, n_test_neg, replace=False)

        # Generate test set
        test_indices = np.concatenate([test_neg_indices, test_pos_indices])

        # The remaining samples are divided for training and validation sets.
        remaining_indices = np.setdiff1d(indices, test_indices)

        # Select 20% of the remaining samples as the validation set.
        p1 = int(np.ceil(0.2 * dataset_size))
        val_indices = np.random.choice(remaining_indices, p1, replace=False)

        # The remaining samples are the training set.
        train_indices = np.setdiff1d(remaining_indices, val_indices)

        # Sort indices
        train_indices.sort()
        val_indices.sort()
        test_indices.sort()

        if dataset == "train":
            sets = ['Training set', 'Validation set', 'Test set']
            indices = [train_indices, val_indices, test_indices]

            data = {
                'Set': sets,
                'Total': [len(idx) for idx in indices],
                'Positive': [csv.iloc[idx][csv.columns[1]].sum() for idx in indices],
                'Negative': [len(idx) - csv.iloc[idx][csv.columns[1]].sum() for idx in indices],
                'GermRate': [f"{(csv.iloc[idx][csv.columns[1]].sum() / len(idx) * 100):.2f}%"
                                        if len(idx) != 0 else "0.00%" for idx in indices]
            }
            df = pd.DataFrame(data)
            print(df.to_string(index=False))

        if dataset == "train":
            return csv.iloc[train_indices]
        elif dataset == "val":
            return csv.iloc[val_indices]
        else:
            return csv.iloc[test_indices]

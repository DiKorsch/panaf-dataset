import os
from typing import Optional
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from dataset.datasets import SupervisedPanAf
from torchvision import transforms
"""
Trainer args (accelerator, devices, num_nodes, etc…)
Data args (sequence length, stride, etc...)
Model specific args (layer_dim, num_layers, learning_rate, etc…)
Program arguments (data_path, cluster_email, etc…)
"""


class SupervisedPanAfDataModule(LightningDataModule):
    def __init__(
        self,
        # Program args
        data_dir: str = None,
        ann_dir: str = None,
        dense_dir: str = None,
        # Data args
        sequence_len: int = 5,
        sample_itvl: int = 1,
        stride: int = 1,
        type: str = None,
        behaviour_threshold: int = 24,
        transform: Optional = None,
        batch_size: int = None,
        num_workers: int = None,
        shuffle: bool = True,
        pin_memory: bool = True,
    ):
        self.data_dir = data_dir
        self.ann_dir = ann_dir
        self.dense_dir = dense_dir
        self.sequence_len = sequence_len
        self.sample_itvl = sample_itvl
        self.stride = stride
        self.type = type
        self.behaviour_threshold = behaviour_threshold
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory

        super().__init__()

    def setup(self, stage: Optional[str] = None):
        # TODO: inc. transforms here
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((244, 244))]
        )
        print(self.type.strip('"'))

        self.train_dataset = SupervisedPanAf(
            data_dir=os.path.join(self.data_dir.strip('"'), "train"),
            ann_dir=os.path.join(self.ann_dir.strip('"'), "train"),
            dense_dir=os.path.join(self.dense_dir.strip('"'), "train"),
            sequence_len=self.sequence_len,
            sample_itvl=self.sample_itvl,
            stride=self.stride,
            type=self.type.strip('"'),
            transform=self.transform,
            behaviour_threshold=self.behaviour_threshold,
        )

        self.validation_dataset = SupervisedPanAf(
            data_dir=os.path.join(self.data_dir.strip('"'), "validation"),
            ann_dir=os.path.join(self.ann_dir.strip('"'), "validation"),
            dense_dir=os.path.join(self.dense_dir.strip('"'), "validation"),
            sequence_len=self.sequence_len,
            sample_itvl=self.sample_itvl,
            stride=self.stride,
            type=self.type.strip('"'),
            transform=self.transform,
            behaviour_threshold=self.behaviour_threshold,
        )

        self.test_dataset = SupervisedPanAf(
            data_dir=os.path.join(self.data_dir.strip('"'), "test"),
            ann_dir=os.path.join(self.ann_dir.strip('"'), "test"),
            dense_dir=os.path.join(self.dense_dir.strip('"'), "test"),
            sequence_len=self.sequence_len,
            sample_itvl=self.sample_itvl,
            stride=self.stride,
            type=self.type.strip('"'),
            transform=self.transform,
            behaviour_threshold=self.behaviour_threshold,
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return train_loader

    def val_dataloader(self):
        validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return validation_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return test_loader

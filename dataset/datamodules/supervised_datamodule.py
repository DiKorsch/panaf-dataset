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
    def __init__(self, cfg):

        self.data_dir = cfg.get("program", "data_dir")
        self.ann_dir = cfg.get("program", "ann_dir")
        self.dense_dir = cfg.get("program", "dense_dir")
        self.sequence_len = cfg.getint("dataset", "sequence_len")
        self.sample_itvl = cfg.getint("dataset", "sample_itvl")
        self.stride = cfg.getint("dataset", "stride")
        self.type = cfg.get("dataset", "type")
        self.behaviour_threshold = cfg.getint("dataset", "behaviour_threshold")
        # self.transform = transform
        self.batch_size = cfg.getint("loader", "batch_size")
        self.num_workers = cfg.getint("loader", "num_workers")
        self.train_shuffle = cfg.getboolean("loader", "train_shuffle")
        self.test_shuffle = cfg.getboolean("loader", "test_shuffle")
        self.pin_memory = cfg.getboolean("loader", "pin_memory")

        super().__init__()

    def setup(self, stage: Optional[str] = None):
        # TODO: inc. transforms here
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((244, 244))]
        )
        print(self.type.strip('"'))

        self.train_dataset = SupervisedPanAf(
            data_dir=os.path.join(self.data_dir, "train"),
            ann_dir=os.path.join(self.ann_dir, "train"),
            dense_dir=os.path.join(self.dense_dir, "train"),
            sequence_len=self.sequence_len,
            sample_itvl=self.sample_itvl,
            stride=self.stride,
            type=self.type,
            transform=self.transform,
            behaviour_threshold=self.behaviour_threshold,
        )

        self.validation_dataset = SupervisedPanAf(
            data_dir=os.path.join(self.data_dir, "validation"),
            ann_dir=os.path.join(self.ann_dir, "validation"),
            dense_dir=os.path.join(self.dense_dir, "validation"),
            sequence_len=self.sequence_len,
            sample_itvl=self.sample_itvl,
            stride=self.stride,
            type=self.type,
            transform=self.transform,
            behaviour_threshold=self.behaviour_threshold,
        )

        self.test_dataset = SupervisedPanAf(
            data_dir=os.path.join(self.data_dir, "test"),
            ann_dir=os.path.join(self.ann_dir, "test"),
            dense_dir=os.path.join(self.dense_dir, "test"),
            sequence_len=self.sequence_len,
            sample_itvl=self.sample_itvl,
            stride=self.stride,
            type=self.type,
            transform=self.transform,
            behaviour_threshold=self.behaviour_threshold,
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return train_loader

    def val_dataloader(self):
        validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=self.test_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return validation_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.test_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return test_loader

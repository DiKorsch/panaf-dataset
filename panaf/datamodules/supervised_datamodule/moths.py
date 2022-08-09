import configparser
import numpy as np
import typing as T

from catalyst.data import BalanceClassSampler
from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms as tr

from panaf.datasets import MothsDataset
from panaf.samplers import BalancedBatchSampler

class SupervisedMothsDataModule(LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        assert cfg.get("remote", "slurm") == "local", \
            "Remote slurm support not implemented yet!"

        root = Path(cfg.get("program", "data_dir"))
        self.images_dir = root / "images"

        assert self.images_dir.exists(), \
            f"Could not find images directory: {self.images_dir}"

        self.val_split_id = cfg.getint("dataset", "val_split_id")
        self.test_split_id = cfg.getint("dataset", "test_split_id")

        self._load_annotations(root)

        self.batch_size = cfg.getint("loader", "batch_size")
        self.num_workers = cfg.getint("loader", "num_workers")
        self.train_shuffle = cfg.getboolean("loader", "train_shuffle")
        self.test_shuffle = cfg.getboolean("loader", "test_shuffle")
        self.pin_memory = cfg.getboolean("loader", "pin_memory")

        try:
            self.which_classes = cfg.get("dataset", "classes")
        except configparser.NoOptionError:
            self.which_classes = None


    def _load_annotations(self, root: Path):

        self._images = np.loadtxt(root / "images.txt",
            dtype=[("id", np.int32), ("fname", "U255")])

        _labels = np.loadtxt(root / "labels.txt",
            dtype=np.int32)

        cls_idxs, self._labels = np.unique(_labels, return_inverse=True)

        self._splits = np.loadtxt(root / "tr_ID.txt",
            dtype=np.int32)


    def setup(self, stage: T.Optional[str] = None):

        self.transform = tr.Compose([
            tr.ToTensor(),
            tr.Resize((244, 244)),
            tr.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        if stage in ("fit", None):
            self.train_ds = self._new_dataset()
            self.val_ds = self._new_dataset(self.val_split_id)

        if stage in ("test", None):
            self.test_ds = self._new_dataset(self.test_split_id)


    def train_dataloader(self):
        return self._new_loader(self.train_ds, self.train_shuffle)

    def val_dataloader(self):
        return self._new_loader(self.val_ds, self.test_shuffle)

    def test_dataloader(self):
        return self._new_loader(self.test_ds, self.test_shuffle)


    def _new_loader(self, dataset, shuffle: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def _new_dataset(self, split_id: T.Optional[int] = None):
        if split_id is not None:
            split_mask = self._splits == split_id
        else:
            val_split = self._splits == self.val_split_id
            test_split = self._splits == self.test_split_id
            split_mask = np.logical_and(~val_split, ~test_split)

        return MothsDataset(
            self.images_dir,
            self._images[split_mask]["fname"],
            self._labels[split_mask],

            transform=self.transform,
        )

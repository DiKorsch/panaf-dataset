import os
from typing import Optional
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from panaf.datasets import SupervisedPanAfPairs
from panaf.samplers import BalancedBatchSampler
from torchvision import transforms
from catalyst.data import BalanceClassSampler
from catalyst.data import DistributedSamplerWrapper
from configparser import NoOptionError

"""
Trainer args (accelerator, devices, num_nodes, etc…)
Data args (sequence length, stride, etc...)
Model specific args (layer_dim, num_layers, learning_rate, etc…)
Program arguments (data_path, cluster_email, etc…)
"""


class SupervisedPanAfPairsDataModule(LightningDataModule):
    def __init__(self, cfg):

        if cfg.get("remote", "slurm") == "ssd":
            self.remote = True
            user = os.getenv("USER")
            slurm_job_id = os.getenv("SLURM_JOB_ID")
            self.path = f"/raid/local_scratch/{user}/{slurm_job_id}"
            data_dir = f"{self.path}/{cfg.get('program', 'data_dir')}"
            ann_dir = f"{self.path}/{cfg.get('program', 'ann_dir')}"
            dense_dir = f"{self.path}/{cfg.get('program', 'dense_dir')}"
            flow_dir = f"{self.path}/{cfg.get('program', 'flow_dir')}"

        elif cfg.get("remote", "slurm") == "hdd":
            self.remote = True
            data_dir = cfg.get("program", "data_dir")
            ann_dir = cfg.get("program", "ann_dir")
            dense_dir = cfg.get("program", "dense_dir")
            flow_dir = cfg.get("program", "flow_dir")

        elif cfg.get("remote", "slurm") == "local":
            self.remote = False
            data_dir = cfg.get("program", "data_dir")
            ann_dir = cfg.get("program", "ann_dir")
            dense_dir = cfg.get("program", "dense_dir")
            flow_dir = cfg.get("program", "flow_dir")

        self.data_dir = data_dir
        self.ann_dir = ann_dir
        self.dense_dir = dense_dir
        self.flow_dir = flow_dir
        self.sequence_len = cfg.getint("dataset", "sequence_len")
        self.sample_itvl = cfg.getint("dataset", "sample_itvl")
        self.stride = cfg.getint("dataset", "stride")
        self.type = cfg.get("dataset", "type")
        self.train_threshold = cfg.getint("dataset", "train_threshold")
        self.test_threshold = cfg.getint("dataset", "test_threshold")

        # self.transform = transform
        self.batch_size = cfg.getint("loader", "batch_size")
        self.num_workers = cfg.getint("loader", "num_workers")
        self.train_shuffle = cfg.getboolean("loader", "train_shuffle")
        self.test_shuffle = cfg.getboolean("loader", "test_shuffle")
        self.pin_memory = cfg.getboolean("loader", "pin_memory")
        self.sampler = cfg.get("loader", "sampler")

        try:
            self.which_classes = cfg.get("dataset", "classes")
        except NoOptionError:
            self.which_classes = None

        super().__init__()

    def setup(self, stage: Optional[str] = None):
        # TODO: inc. transforms here
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((244, 244))]
        )

        if stage == "fit" or stage is None:

            self.train_dataset = SupervisedPanAfPairs(
                data_dir=os.path.join(self.data_dir, "train"),
                ann_dir=os.path.join(self.ann_dir, "train"),
                dense_dir=os.path.join(self.dense_dir, "train"),
                flow_dir=self.flow_dir,
                sequence_len=self.sequence_len,
                sample_itvl=self.sample_itvl,
                stride=self.stride,
                type=self.type,
                transform=self.transform,
                behaviour_threshold=self.train_threshold,
                which_classes=self.which_classes,
            )

            self.validation_dataset = SupervisedPanAfPairs(
                data_dir=os.path.join(self.data_dir, "validation"),
                ann_dir=os.path.join(self.ann_dir, "validation"),
                dense_dir=os.path.join(self.dense_dir, "validation"),
                flow_dir=self.flow_dir,
                sequence_len=self.sequence_len,
                sample_itvl=self.sample_itvl,
                stride=self.stride,
                type=self.type,
                transform=self.transform,
                behaviour_threshold=self.test_threshold,
                which_classes=self.which_classes,
            )

        if stage == "test" or stage is None:

            self.test_dataset = SupervisedPanAfPairs(
                data_dir=os.path.join(self.data_dir, "test"),
                ann_dir=os.path.join(self.ann_dir, "test"),
                dense_dir=os.path.join(self.dense_dir, "test"),
                flow_dir=self.flow_dir,
                sequence_len=self.sequence_len,
                sample_itvl=self.sample_itvl,
                stride=self.stride,
                type=self.type,
                transform=self.transform,
                behaviour_threshold=self.test_threshold,
            )

        # Configure based on cfg
        self.configure_sampler()

    def configure_sampler(self):
        if self.sampler == "upsampling":
            self.sampler = BalanceClassSampler(
                labels=self.train_dataset.targets, mode="upsampling"
            )
            self.train_shuffle = False

        elif self.sampler == "downsampling":
            self.sampler = BalanceClassSampler(
                labels=self.train_dataset.targets, mode="downsampling"
            )
            self.train_shuffle = False

        elif self.sampler == "balanced":
            self.sampler = BalancedBatchSampler(
                self.train_dataset, self.train_dataset.targets
            )
            self.train_shuffle = False
        else:
            self.sampler = None

        if self.remote and self.sampler is not None:
            self.sampler = DistributedSamplerWrapper(self.sampler)

    def get_logit_adjustments(self):
        return self.train_dataset.compute_logit_adjustment()

    def train_dataloader(self):

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            sampler=self.sampler,
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

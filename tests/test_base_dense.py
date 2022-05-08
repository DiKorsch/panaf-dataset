import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
from dataset import PanAfDataset


class TestDense:

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "tests/data/single/videos"
    ann_dir = "tests/data/single/annotations"
    dense_dir = "tests/data/single/annotations"

    def test_check_dense_exists_true(self):
        """Test 5-frame sequence."""
        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            stride=5,
            transform=self.transform,
        )

        with open(
            "tests/data/cases/dense/example_two/annotations/0YvgQsXboK.pkl", "rb"
        ) as handle:
            ann = pickle.load(handle)

        dense = dataset.check_dense_exists(ann=ann, frame_no=1, current_ape=0)

        assert dense

    def test_check_dense_exists_false(self):
        """Test 5-frame sequence."""
        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            stride=5,
            transform=self.transform,
        )

        with open(
            "tests/data/cases/dense/example_two/annotations/0YvgQsXboK.pkl", "rb"
        ) as handle:
            ann = pickle.load(handle)

        dense = dataset.check_dense_exists(ann=ann, frame_no=82, current_ape=0)

        assert not dense

    def test_get_dense_annotation(self):

        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            dense_dir=self.dense_dir,
            sequence_len=5,
            sample_itvl=1,
            stride=5,
            transform=self.transform,
        )

        ann = dataset.get_dense_annotation("0YvgQsXboK")

        assert ann["video"] == "0YvgQsXboK"

    def test_build_spatial_sample(self):
        # TODO: move this to test_base.py
        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            dense_dir=self.dense_dir,
            sequence_len=5,
            sample_itvl=1,
            stride=5,
            type="r",
            transform=self.transform,
        )

        dataloader = DataLoader(dataset)
        sample = next(iter(dataloader))
        assert len(sample["spatial_sample"].squeeze(dim=0)) == 5

    def test_build_dense_sample(self):

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((244, 244))]
        )

        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            dense_dir=self.dense_dir,
            sequence_len=5,
            sample_itvl=1,
            stride=5,
            type="rd",
            transform=transform,
        )

        dataloader = DataLoader(dataset)
        sample = next(iter(dataloader))

        print(sample.keys())

        assert len(sample["spatial_sample"].squeeze(dim=0)) == 5
        assert len(sample["dense_sample"].squeeze(dim=0)) == 5

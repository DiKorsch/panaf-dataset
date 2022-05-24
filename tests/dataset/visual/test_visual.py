import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
from panaf.datasets import PanAfDataset
from panaf.datasets import SupervisedPanAf


class TestDense:

    transform = transforms.Compose([transforms.Resize((244, 244))])

    data_dir = "tests/data/single/videos"
    ann_dir = "tests/data/single/annotations"
    dense_dir = "tests/data/single/annotations"
    flow_dir = "tests/data/single/flow"

    def test_small_sequence(self):
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
            "tests/data/cases/dense/example_one/annotations/0hu96Jv2As_dense.pkl", "rb"
        ) as handle:
            ann = pickle.load(handle)

        dense_sample = dataset.build_dense_sample(ann, "0hu96Jv2As", 0, 1)
        fig = plt.figure(figsize=(8, 6))
        grid = make_grid(dense_sample).permute(1, 2, 0)
        plt.imshow(grid)
        plt.show()

        assert len(dense_sample) == 5

    def test_dense_sequence(self):
        """Test 5-frame sequence."""
        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=8,
            sample_itvl=10,
            stride=5,
            transform=self.transform,
        )
        with open(
            "tests/data/cases/dense/example_two/annotations/0YvgQsXboK_dense.pkl", "rb"
        ) as handle:
            ann = pickle.load(handle)

        dense_sample = dataset.build_dense_sample(ann, "0YvgQsXboK", 0, 1)
        fig = plt.figure(figsize=(8, 6))
        dense_sample = list(reversed(dense_sample))
        grid = make_grid(dense_sample).permute(1, 2, 0)
        plt.imshow(grid)
        plt.show()

        assert len(dense_sample) == 8

    def test_sample_itvl(self):
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
            "tests/data/cases/dense/example_two/annotations/0YvgQsXboK_dense.pkl", "rb"
        ) as handle:
            ann = pickle.load(handle)

        dense_sample = dataset.build_dense_sample(ann, "0YvgQsXboK", 0, 1)
        fig = plt.figure(figsize=(8, 6))
        dense_sample = list(reversed(dense_sample))
        grid = make_grid(dense_sample).permute(1, 2, 0)
        plt.imshow(grid)
        plt.show()

        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=10,
            stride=50,
            transform=self.transform,
        )

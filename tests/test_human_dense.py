import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
from dataset import SupervisedPanAf


class TestDenseInstantiation:

    transform = transforms.Compose([transforms.Resize((244, 244))])

    data_dir = "tests/data/cases/dense/example_three/videos/"
    dense_dir = "tests/data/cases/dense/example_three/annotations/dense"
    std_dir = "tests/data/cases/dense/example_three/annotations/std"

    def test_dense_instantiation(self):
        """Test 5-frame sequence."""
        dense_dataset = SupervisedPanAf(
            data_dir=self.data_dir,
            ann_dir=self.dense_dir,
            sequence_len=5,
            sample_itvl=1,
            stride=5,
            type='rd',
            transform=self.transform,
        )

        dataset = SupervisedPanAf(
            data_dir=self.data_dir,
            ann_dir=self.std_dir,
            sequence_len=5,
            sample_itvl=1,
            stride=5,
            type='r',
            transform=self.transform,
        )

        assert dataset.__len__() != dense_dataset.__len__()

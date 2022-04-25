import pytest
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
from dataset import PanAfDataset


class Test:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "data/videos"
    ann_dir = "data/annotations"

    dataset = PanAfDataset(
        data_dir=data_dir,
        ann_dir=ann_dir,
        sequence_len=10,
        sample_itvl=1,
        transform=transform,
    )

    def test_samples(self):
        assert self.dataset.__len__() == 8

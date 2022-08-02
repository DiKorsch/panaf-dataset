"""
Summary: human class tests.
Tests on sample containing 1 gorilla displaying 'walking'
behaviour for 86 consecutive frames (frames 1 - 86).
"""
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from panaf.datasets import SupervisedPanAf
from torchvision.utils import make_grid


class TestSupervisedFlow:

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "tests/data/single/videos"
    ann_dir = "tests/data/single/annotations"
    flow_dir = "tests/data/single/flow"

    def test_flow_sequence(self):
        """Test 5-frame sequence with 5-frame behaviour thresh."""
        dataset = SupervisedPanAf(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            dense_dir=self.ann_dir,
            flow_dir=self.flow_dir,
            sequence_len=5,
            sample_itvl=3,
            stride=5,
            type="rf",
            spatial_transform=self.transform,
            temporal_transform=self.transform,
            behaviour_threshold=24,
        )
        loader = DataLoader(dataset)
        sample, label = next(iter(loader))
        assert len(sample["flow_sample"].squeeze()) == 5

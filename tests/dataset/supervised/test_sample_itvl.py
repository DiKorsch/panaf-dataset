"""
Summary: human class tests.
Tests on sample containing 1 gorilla displaying 'walking'
behaviour for 86 consecutive frames (frames 1 - 86).
"""

from torchvision import transforms
from torch.utils.data import DataLoader
from panaf.datasets import SupervisedPanAf


class TestSampleItvl:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "tests/data/single/videos"
    ann_dir = "tests/data/single/annotations"

    def test_sample_itvl(self):
        """Test 5-frame sequence with 5-frame behaviour thresh."""
        dataset = SupervisedPanAf(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            dense_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=3,
            stride=5,
            type="r",
            transform=self.transform,
            behaviour_threshold=24,
        )
        print(dataset.samples)
        loader = DataLoader(dataset)
        for x, y in loader:
            pass

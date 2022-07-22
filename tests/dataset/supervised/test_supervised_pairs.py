"""
Summary: human class tests.
Tests on sample containing 1 gorilla displaying 'walking'
behaviour for 86 consecutive frames (frames 1 - 86).
"""

from torchvision import transforms
from torch.utils.data import DataLoader
from panaf.datasets import SupervisedPanAfPairs


class TestSingleApe:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "tests/data/single/videos"
    ann_dir = "tests/data/single/annotations"

    def test_low_threshold(self):
        """Test 5-frame sequence with 5-frame behaviour thresh."""
        dataset = SupervisedPanAfPairs(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            type='r',
            transform=self.transform,
            behaviour_threshold=5,
        )

        loader = DataLoader(dataset)
        a, p, y = next(iter(loader))

"""
Human class tests: ...
Tests on sample containing 2 apes displaying 'walking'
and 'standing' behaviours. Both apes are in frame
until frame 101. After that the other ape is in frame
until frame 237.
"""

from torchvision import transforms
from dataset import SupervisedPanAf


class TestMultipleApes:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "data/multiple/videos"
    ann_dir = "data/multiple/annotations"

    def test_low_threshold(self):
        """Test 5-frame sequence with 5-frame behaviour thresh."""
        dataset = SupervisedPanAf(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            transform=self.transform,
            behaviour_threshold=5,
        )
        dataset.print_samples()
        assert dataset.__len__() == 65

"""
Human class tests: ...
Tests on sample containing 2 apes displaying 'walking'
and 'standing' behaviours. Both apes are in frame
until frame 101. After that the other ape is in frame
until frame 237.
"""

from torchvision import transforms
from dataset import PanAfDataset


class TestMultipleApes:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "tests/data/multiple/videos"
    ann_dir = "tests/data/multiple/annotations"

    def test_low_threshold(self):
        """Test 5-frame sequence with 5-frame behaviour thresh."""
        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            transform=self.transform,
        )
        assert dataset.__len__() == 67

    def test_count_apes(self):
        """Test dataset.verify_ape_ids() method."""
        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            transform=self.transform,
        )

        # Test annotation with 2 apes
        filename = "1oZGHFgDpe"
        ann = dataset.load_annotation(filename)
        assert dataset.count_apes(ann) == 1

    def test_get_ape_behaviour(self):
        """Test dataset.verify_ape_ids() method."""
        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            transform=self.transform,
        )

        # Test annotation with 1 ape
        filename = "1oZGHFgDpe"
        ann = dataset.load_annotation(filename)
        behaviour_ape_0 = dataset.get_ape_behaviour(ann=ann, current_ape=0, frame_no=10)
        behaviour_ape_1 = dataset.get_ape_behaviour(ann=ann, current_ape=1, frame_no=10)
        assert behaviour_ape_0 == "walking"
        assert behaviour_ape_1 == "standing"

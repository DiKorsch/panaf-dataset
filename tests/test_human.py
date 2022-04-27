"""
Summary: human class tests.
Tests on sample containing 1 gorilla displaying 'walking'
behaviour for 86 consecutive frames (frames 1 - 86).
"""

from torchvision import transforms
from dataset import SupervisedPanAf


class TestSingleApe:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "tests/data/single/videos"
    ann_dir = "tests/data/single/annotations"

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

        # 86 / 5 == 17.2 (i.e., 17 5-frame samples)
        assert dataset.__len__() == 17

    def test_mid_threshold(self):

        dataset = SupervisedPanAf(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            transform=self.transform,
            behaviour_threshold=72,
        )

        assert dataset.__len__() == 3

    def test_mid_threshold_w_stride(self):

        dataset = SupervisedPanAf(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            stride=24,
            transform=self.transform,
            behaviour_threshold=72,
        )

        assert dataset.__len__() == 1

    def test_high_threshold(self):
        dataset = SupervisedPanAf(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            transform=self.transform,
            behaviour_threshold=100,
        )

        assert dataset.__len__() == 0

    def test_check_behaviour_threshold(self):

        dataset = SupervisedPanAf(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            transform=self.transform,
            behaviour_threshold=1,
        )

        filename = "0YvgQsXboK"
        ann = dataset.load_annotation(filename)
        dataset.set_behaviour_threshold(value=5)

        assert dataset.check_behaviour_threshold(
            ann=ann, current_ape=0, frame_no=1, target_behaviour="walking"
        )
        assert not dataset.check_behaviour_threshold(
            ann=ann, current_ape=0, frame_no=85, target_behaviour="walking"
        )

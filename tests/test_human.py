"""
Summary: human class tests.
Tests on sample containing 1 gorilla displaying 'walking'
behaviour for 86 consecutive frames (frames 1 - 86).
"""

from torchvision import transforms
from torch.utils.data import DataLoader
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

    def test_low_threshold_w_stride5(self):
        """Test 5-frame sequence with 5-frame behaviour thresh."""
        dataset = SupervisedPanAf(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            stride=10,
            transform=self.transform,
            behaviour_threshold=5,
        )

        # 86 / 10 == 8.6 (i.e., 8 5-frame samples)
        assert dataset.__len__() == 8

        loader = DataLoader(dataset)
        sequence, behaviour = next(iter(loader))
        assert len(sequence.squeeze(dim=0)) == 5

    def test_low_threshold_w_stride10_itvl4(self):
        """Test 5-frame sequence with 5-frame behaviour thresh."""
        dataset = SupervisedPanAf(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=4,
            stride=10,
            transform=self.transform,
            behaviour_threshold=5,
        )

        assert dataset.__len__() == 8

        loader = DataLoader(dataset)
        sequence, behaviour = next(iter(loader))
        assert len(sequence.squeeze(dim=0)) == 5

    def test_mid_threshold(self):

        dataset = SupervisedPanAf(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            transform=self.transform,
            behaviour_threshold=72,
        )

        assert dataset.__len__() == 17

    def test_mid_threshold_w_seqlen10(self):

        dataset = SupervisedPanAf(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=10,
            sample_itvl=1,
            transform=self.transform,
            behaviour_threshold=72,
        )

        assert dataset.__len__() == 8
        loader = DataLoader(dataset)
        sequence, behaviour = next(iter(loader))
        assert len(sequence.squeeze(dim=0)) == 10
        assert behaviour == 8

    def test_mid_threshold_w_seqlen20(self):

        dataset = SupervisedPanAf(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=20,
            sample_itvl=1,
            transform=self.transform,
            behaviour_threshold=72,
        )

        assert dataset.__len__() == 4
        loader = DataLoader(dataset)
        sequence, behaviour = next(iter(loader))
        assert len(sequence.squeeze(dim=0)) == 20
        assert behaviour == 8

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

        assert dataset.__len__() == 3

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

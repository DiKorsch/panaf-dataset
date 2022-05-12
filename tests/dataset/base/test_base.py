"""
Summary: base class tests.
Tests on sample containing 1 gorilla present for
86 consecutive frames (frames 1 - 86).
"""

from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.datasets import PanAfDataset


class TestSingleApe:

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "tests/data/single/videos"
    ann_dir = "tests/data/single/annotations"

    def test_small_sequence(self):
        """Test 5-frame sequence."""
        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            stride=5,
            type="r",
            transform=self.transform,
        )

        dataloader = DataLoader(dataset)
        sequence = next(iter(dataloader))
        assert dataset.__len__() == 17
        assert len(sequence["spatial_sample"].squeeze(dim=0)) == 5

    def test_mid_sequence(self):
        """Test 10-frame sequence."""
        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=10,
            sample_itvl=1,
            type="r",
            transform=self.transform,
        )

        dataloader = DataLoader(dataset)
        sequence = next(iter(dataloader))
        assert dataset.__len__() == 8
        assert len(sequence["spatial_sample"].squeeze(dim=0)) == 10

    def test_mid_with_stride(self):
        """Test 10-frame sequence w/ stride 20."""
        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=10,
            sample_itvl=1,
            stride=20,
            type="r",
            transform=self.transform,
        )

        dataloader = DataLoader(dataset)
        sequence = next(iter(dataloader))
        assert dataset.__len__() == 4
        assert len(sequence["spatial_sample"].squeeze(dim=0)) == 10

    def test_mid_with_interval(self):
        """Test 10-frame sequence w/ itvl 2."""
        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=2,
            stride=10,
            type="r",
            transform=self.transform,
        )

        dataloader = DataLoader(dataset)
        sequence = next(iter(dataloader))
        assert dataset.__len__() == 8
        assert len(sequence["spatial_sample"].squeeze(dim=0)) == 5

    def test_large_seqence(self):
        """Test sequence of 86-frames (i.e. match total frames)."""
        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=86,
            sample_itvl=1,
            type="r",
            transform=self.transform,
        )

        dataloader = DataLoader(dataset)
        sequence = next(iter(dataloader))
        assert dataset.__len__() == 1
        assert len(sequence["spatial_sample"].squeeze(dim=0)) == 86

    def test_above_thresh_seqence(self):
        """Test sequence 87-frames (i.e. greater than total frames)."""
        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=87,
            sample_itvl=1,
            type="r",
            transform=self.transform,
        )
        assert dataset.__len__() == 0

    def test_verify_ape_ids(self):
        """Test dataset.verify_ape_ids() method."""
        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            transform=self.transform,
        )

        # True example
        ape_no = 3
        ape_ids = [0, 1, 2, 3]

        assert dataset.verify_ape_ids(ape_no, ape_ids)

        # False example
        ape_no = 3
        ape_ids = [0, 1, 2]
        assert not dataset.verify_ape_ids(ape_no, ape_ids)

    def test_count_apes(self):
        """Test dataset.verify_ape_ids() method."""
        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            transform=self.transform,
        )

        # Test annotation with 1 ape
        filename = "0YvgQsXboK"
        ann = dataset.load_annotation(filename)
        assert dataset.count_apes(ann) == 0

    def test_get_video(self):
        """Test dataset.verify_ape_ids() method."""
        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            transform=self.transform,
        )

        # Test annotation with 1 ape
        video = dataset.get_video("foobar")
        assert video is None

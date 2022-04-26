"""
Summary: base class tests.
Tests on sample containing 1 gorilla displaying 'walking'
behaviour for 86 consecutive frames (frames 1 - 86).
"""

from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import PanAfDataset


def test_small_sequence():
    """ Test 5-frame sequence. """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "data/videos"
    ann_dir = "data/annotations"

    dataset = PanAfDataset(
        data_dir=data_dir,
        ann_dir=ann_dir,
        sequence_len=5,
        sample_itvl=1,
        stride=5,
        transform=transform,
    )

    dataloader = DataLoader(dataset)
    sequence = next(iter(dataloader))
    assert dataset.__len__() == 17
    assert len(sequence.squeeze(dim=0)) == 5


def test_mid_sequence():
    """ Test 10-frame sequence. """
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

    dataloader = DataLoader(dataset)
    sequence = next(iter(dataloader))
    assert dataset.__len__() == 8
    assert len(sequence.squeeze(dim=0)) == 10


def test_mid_with_stride():
    """ Test 10-frame sequence w/ stride 20. """
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
        stride=20,
        transform=transform,
    )

    dataloader = DataLoader(dataset)
    sequence = next(iter(dataloader))
    assert dataset.__len__() == 4
    assert len(sequence.squeeze(dim=0)) == 10


def test_mid_with_interval():
    """ Test 10-frame sequence w/ itvl 2. """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "data/videos"
    ann_dir = "data/annotations"

    dataset = PanAfDataset(
        data_dir=data_dir,
        ann_dir=ann_dir,
        sequence_len=5,
        sample_itvl=2,
        stride=10,
        transform=transform,
    )

    dataloader = DataLoader(dataset)
    sequence = next(iter(dataloader))
    assert dataset.__len__() == 8
    assert len(sequence.squeeze(dim=0)) == 5


def test_large_seqence():
    """ Test sequence of 86-frames (i.e. match total frames)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "data/videos"
    ann_dir = "data/annotations"

    dataset = PanAfDataset(
        data_dir=data_dir,
        ann_dir=ann_dir,
        sequence_len=86,
        sample_itvl=1,
        transform=transform,
    )

    dataloader = DataLoader(dataset)
    sequence = next(iter(dataloader))
    assert dataset.__len__() == 1
    assert len(sequence.squeeze(dim=0)) == 86


def test_above_thresh_seqence():
    """ Test sequence 87-frames (i.e. greater than total frames). """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "data/videos"
    ann_dir = "data/annotations"

    dataset = PanAfDataset(
        data_dir=data_dir,
        ann_dir=ann_dir,
        sequence_len=87,
        sample_itvl=1,
        transform=transform,
    )

    assert dataset.__len__() == 0

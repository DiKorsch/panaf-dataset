from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import PanAfDataset

# TODO: baseclass - between sample stride


def test_small_sequence():
    """
    Summary: test 5-frame sequence
    Test sample contains 1 gorilla displaying 'walking'
    behaviour for 86 consecutive frames (frames 1 - 86).
    """
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
        transform=transform,
    )

    dataloader = DataLoader(dataset)
    sequence = next(iter(dataloader))
    assert dataset.__len__() == 17
    assert len(sequence.squeeze(dim=0)) == 5


def test_mid_sequence():
    """
    Summary: test 10-frame sequence
    Test sample contains 1 gorilla displaying 'walking'
    behaviour for 86 consecutive frames (frames 1 - 86).
    """
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


def test_mid_with_interval():
    """
    Summary: test 5-frame sequence with itvl=2 (i.e., 10-frame window)
    Test sample contains 1 gorilla displaying 'walking'
    behaviour for 86 consecutive frames (frames 1 - 86).
    """
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
        transform=transform,
    )

    dataloader = DataLoader(dataset)
    sequence = next(iter(dataloader))
    assert dataset.__len__() == 16
    assert len(sequence.squeeze(dim=0)) == 5


def test_large_seqence():
    """
    Summary: test small sample
    Test sample contains 1 gorilla displaying 'walking'
    behaviour for 86 consecutive frames (frames 1 - 86).
    """
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
    """
    Summary: test with sequence len above thresh
    Test sample contains 1 gorilla displaying 'walking'
    behaviour for 86 consecutive frames (frames 1 - 86).
    """
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
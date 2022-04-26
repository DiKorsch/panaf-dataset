from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import PanAfDataset
from dataset import SupervisedPanAf


def test_high_threshold():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "data/videos"
    ann_dir = "data/annotations"

    dataset = SupervisedPanAf(
        data_dir=data_dir,
        ann_dir=ann_dir,
        sequence_len=5,
        sample_itvl=1,
        transform=transform,
        behaviour_threshold=100,
    )

    dataloader = DataLoader(dataset)
    inputs, behaviour = next(iter(dataloader))

    assert dataset.__len__() == 0


def test_mid_threshold():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "data/videos"
    ann_dir = "data/annotations"

    dataset = SupervisedPanAf(
        data_dir=data_dir,
        ann_dir=ann_dir,
        sequence_len=5,
        sample_itvl=1,
        transform=transform,
        behaviour_threshold=70,
    )

    assert dataset.__len__() == 4


def test_low_threshold():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "data/videos"
    ann_dir = "data/annotations"

    dataset = SupervisedPanAf(
        data_dir=data_dir,
        ann_dir=ann_dir,
        sequence_len=5,
        sample_itvl=1,
        transform=transform,
        behaviour_threshold=5,
    )

    assert dataset.__len__() == 17


def test_supervised():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "data/videos"
    ann_dir = "data/annotations"

    dataset = SupervisedPanAf(
        data_dir=data_dir,
        ann_dir=ann_dir,
        sequence_len=10,
        sample_itvl=1,
        transform=transform,
        behaviour_threshold=100,
    )

    assert dataset.__len__() == 0


class Test:
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

    def test_samples(self):
        assert self.dataset.__len__() == 8

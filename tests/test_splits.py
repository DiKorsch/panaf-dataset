"""
Summary: test instantiation of splits.
"""

from torchvision import transforms
from dataset.datasets import SupervisedPanAf


def test_train_split():
    """Test 5-frame sequence with 5-frame behaviour thresh."""

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "/home/dl18206/Desktop/phd/data/panaf/obfuscated/restructure/data/train"
    ann_dir = (
        "/home/dl18206/Desktop/phd/data/panaf/obfuscated/restructure/annotations/train"
    )

    dataset = SupervisedPanAf(
        data_dir=data_dir,
        ann_dir=ann_dir,
        sequence_len=5,
        sample_itvl=1,
        stride=5,
        transform=transform,
        behaviour_threshold=24,
    )

    assert dataset.count_videos() == 400
    print(f"Training samples: {dataset.__len__()}")
    dataset.print_samples_by_class()


def test_val_split():
    """Test 5-frame sequence with 5-frame behaviour thresh."""

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = (
        "/home/dl18206/Desktop/phd/data/panaf/obfuscated/restructure/data/validation"
    )
    ann_dir = "/home/dl18206/Desktop/phd/data/panaf/obfuscated/restructure/annotations/validation"

    dataset = SupervisedPanAf(
        data_dir=data_dir,
        ann_dir=ann_dir,
        sequence_len=5,
        sample_itvl=1,
        stride=5,
        transform=transform,
        behaviour_threshold=24,
    )

    assert dataset.count_videos() == 25
    print(f"Validation samples: {dataset.__len__()}")
    dataset.print_samples_by_class()

def test_test_split():
    """Test 5-frame sequence with 5-frame behaviour thresh."""

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "/home/dl18206/Desktop/phd/data/panaf/obfuscated/restructure/data/test"
    ann_dir = (
        "/home/dl18206/Desktop/phd/data/panaf/obfuscated/restructure/annotations/test"
    )

    dataset = SupervisedPanAf(
        data_dir=data_dir,
        ann_dir=ann_dir,
        sequence_len=5,
        sample_itvl=1,
        stride=5,
        transform=transform,
        behaviour_threshold=24,
    )

    assert dataset.count_videos() == 75
    print(f"Test samples: {dataset.__len__()}")
    dataset.print_samples_by_class()

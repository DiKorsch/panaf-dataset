"""
Summary: test instantiation of splits.
"""

from torchvision import transforms
from panaf.datasets import SupervisedPanAf


def test_train_split():
    """Test 5-frame sequence with 5-frame behaviour thresh."""

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "/home/dl18206/Desktop/phd/data/panaf/obfuscated/restructure/data/train"
    ann_dir = "/home/dl18206/Desktop/phd/data/panaf/obfuscated/restructure/annotations/standard/train"

    dataset = SupervisedPanAf(
        data_dir=data_dir,
        ann_dir=ann_dir,
        sequence_len=5,
        sample_itvl=1,
        stride=5,
        transform=transform,
        behaviour_threshold=24,
        which_classes="minority",
    )

    print(f"Training samples: {dataset.__len__()}")
    dataset.print_samples_by_class()
    print(f"Classes: {dataset.classes}")


def test_val_split():
    """Test 5-frame sequence with 5-frame behaviour thresh."""

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = (
        "/home/dl18206/Desktop/phd/data/panaf/obfuscated/restructure/data/validation"
    )
    ann_dir = "/home/dl18206/Desktop/phd/data/panaf/obfuscated/restructure/annotations/standard/validation"

    dataset = SupervisedPanAf(
        data_dir=data_dir,
        ann_dir=ann_dir,
        sequence_len=5,
        sample_itvl=1,
        stride=5,
        transform=transform,
        behaviour_threshold=5,
        which_classes="minority",
    )

    print(f"Validation samples: {dataset.__len__()}")
    dataset.print_samples_by_class()
    print(f"Classes: {dataset.classes}")


def test_test_split():
    """Test 5-frame sequence with 5-frame behaviour thresh."""

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "/home/dl18206/Desktop/phd/data/panaf/obfuscated/restructure/data/test"
    ann_dir = "/home/dl18206/Desktop/phd/data/panaf/obfuscated/restructure/annotations/standard/test"

    dataset = SupervisedPanAf(
        data_dir=data_dir,
        ann_dir=ann_dir,
        sequence_len=5,
        sample_itvl=1,
        stride=5,
        transform=transform,
        behaviour_threshold=5,
        which_classes="minority",
    )

    print(f"Test samples: {dataset.__len__()}")
    dataset.print_samples_by_class()
    print(f"Classes: {dataset.classes}")

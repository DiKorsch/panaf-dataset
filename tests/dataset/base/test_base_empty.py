"""
Base class tests: ...
Tests on sample with no apes detected in any frames.
"""

from torchvision import transforms
from panaf.datasets import PanAfDataset


class TestEmpty:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "tests/data/cases/empty"
    ann_dir = "tests/data/cases/empty"

    def test_count_apes(self):
        """Test dataset.count_apes() method."""
        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            transform=self.transform,
        )

        # Test annotation with no apes
        filename = "ACP0000b9x"
        ann = dataset.load_annotation(filename)
        assert dataset.count_apes(ann) is None

    def test_get_ape_coords(self):
        """Test dataset.count_apes() method."""
        dataset = PanAfDataset(
            data_dir=self.data_dir,
            ann_dir=self.ann_dir,
            sequence_len=5,
            sample_itvl=1,
            transform=self.transform,
        )

        # Test annotation with no apes
        filename = "ACP0000b9x"
        coords = dataset.get_ape_coords(video=filename, ape_id=0, frame_idx=1)
        assert coords is None

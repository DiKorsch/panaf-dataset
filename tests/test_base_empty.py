"""
Base class tests: ...
Tests on sample with no apes detected in any frames.
"""

from torchvision import transforms
from dataset import PanAfDataset


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
        assert not dataset.count_apes(ann)

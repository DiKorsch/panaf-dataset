import pickle
from torchvision import transforms
from dataset.datasets import SupervisedPanAf
from torch.utils.data import DataLoader


class TestDenseInstantiation:

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    data_dir = "tests/data/cases/dense/error_five/data"
    dense_dir = "tests/data/cases/dense/error_five/annotations"

    def test_dense_instantiation(self):
        """Test 5-frame sequence."""
        dense_dataset = SupervisedPanAf(
            data_dir=self.data_dir,
            ann_dir=self.dense_dir,
            dense_dir=self.dense_dir,
            sequence_len=5,
            sample_itvl=1,
            stride=5,
            type="rd",
            transform=self.transform,
            behaviour_threshold=24,
        )

        loader = DataLoader(dense_dataset, shuffle=False, batch_size=1)
        for sample, label in loader:
            pass

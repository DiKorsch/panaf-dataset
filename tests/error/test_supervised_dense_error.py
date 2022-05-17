import pickle
from torchvision import transforms
from panaf.datasets import SupervisedPanAf
from torch.utils.data import DataLoader


class TestDenseInstantiation:

    transform = transforms.Compose([transforms.Resize((244, 244))])

    data_dir = "tests/data/cases/dense/error/data/train"
    dense_dir = "tests/data/cases/dense/error/annotations/train"

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

        with open(
            "tests/data/cases/dense/error/annotations/train/D5D1IoY7jA_dense.pkl", "rb"
        ) as handle:
            ann = pickle.load(handle)

        valid_frames = dense_dataset.get_valid_frames(ann, 1, "standing", 132, 360)

        loader = DataLoader(dense_dataset, batch_size=1)
        for sample, label in loader:
            pass

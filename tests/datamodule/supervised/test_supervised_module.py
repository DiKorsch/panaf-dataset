import configparser
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from dataset.datamodules import SupervisedPanAfDataModule


class TestConfig:

    cfg = configparser.ConfigParser()
    cfg.read("tests/data/config/config.cfg")

    def test_module_initialisation(self):

        data_module = SupervisedPanAfDataModule(
            data_dir=self.cfg["program"]["data_dir"],
            ann_dir=self.cfg["program"]["ann_dir"],
            dense_dir=self.cfg["program"]["dense_dir"],
            sequence_len=int(self.cfg["dataset"]["sequence_len"]),
            sample_itvl=int(self.cfg["dataset"]["sample_itvl"]),
            stride=int(self.cfg["dataset"]["stride"]),
            type=self.cfg["dataset"]["type"],
            behaviour_threshold=int(self.cfg["dataset"]["behaviour_threshold"]),
            batch_size=int(self.cfg["loader"]["batch_size"]),
            num_workers=int(self.cfg["loader"]["num_workers"]),
            shuffle=bool(self.cfg["loader"]["shuffle"]),
            pin_memory=bool(self.cfg["loader"]["pin_memory"]),
        )

        data_module.setup()
        sample, behaviour = next(iter(data_module.train_dataloader()))
        assert len(sample["spatial_sample"].squeeze()) == int(
            self.cfg["dataset"]["sequence_len"]
        )

        fig = plt.figure(figsize=(8, 6))
        grid = make_grid(sample["dense_sample"].squeeze()).permute(1, 2, 0)
        plt.imshow(grid)
        plt.show()

import configparser
from ast import literal_eval
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from panaf.datamodules import SupervisedPanAfDataModule


class TestConfig:

    cfg = configparser.ConfigParser()
    cfg.read("tests/data/config/config.cfg")

    def test_module_initialisation(self):

        data_module = SupervisedPanAfDataModule(cfg=self.cfg)

        data_module.setup()

        sample, behaviour = next(iter(data_module.train_dataloader()))

        assert len(sample["spatial_sample"].squeeze()) == int(
            self.cfg["dataset"]["sequence_len"]
        )

        fig = plt.figure(figsize=(8, 6))
        grid = make_grid(sample["dense_sample"].squeeze()).permute(1, 2, 0)
        plt.imshow(grid)
        plt.show()

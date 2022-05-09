import configparser


class TestConfig:

    cfg = configparser.ConfigParser()
    cfg.read("tests/data/config/config.cfg")

    def test_config(self):
        assert int(self.cfg["data"]["sequence_length"]) == 5

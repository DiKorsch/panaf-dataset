from dataset.base import PanAfDataset


class PanAfHumanDataset(PanAfDataset):
    def __init__(self, data_dir, ann_dir, sequence_len, transform):
        super().__init__(data_dir, ann_dir, sequence_len, transform)

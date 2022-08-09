import numpy as np

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class MothsDataset(Dataset):

    def __init__(self, root: Path, images, labels, *, transform):
        super().__init__()
        self._root = root
        self._images = images
        self._labels = labels

        assert callable(transform), "Transform should be callable!"
        self.transform = transform

    def _read_image(self, im_path):
        with Image.open(im_path) as im:
            return np.array(im.convert("RGB"))

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, i: int):
        im_path = self._images[i]
        lab = self._labels[i]

        im = self._read_image(self._root / im_path)
        if self.transform is not None:
            im = self.transform(im)

        return im, lab

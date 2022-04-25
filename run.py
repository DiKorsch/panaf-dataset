import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
from dataset.base import PanAfDataset
from dataset.human import PanAfHumanDataset


def main():

    print("=> Base class")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    dataset = PanAfDataset(
        data_dir="/home/dl18206/Desktop/phd/data/panaf/acp/videos",
        ann_dir="/home/dl18206/Desktop/phd/data/panaf/acp/annotations/machine/json/all/long",
        sequence_len=8,
        transform=transform,
    )

    print("=> Human Dataset")

    dataset = PanAfHumanDataset(
        data_dir="/home/dl18206/Desktop/phd/data/panaf/acp/videos",
        ann_dir="/home/dl18206/Desktop/phd/data/panaf/acp/annotations/machine/json/all/long",
        sequence_len=8,
        transform=transform,
    )

    print(dataset.__len__())

    dataloader = DataLoader(dataset)
    inputs = next(iter(dataloader))

    print(inputs.shape)
    print(inputs.squeeze(dim=0).shape)

    fig = plt.figure(figsize=plt.figaspect(0.75))
    inputs = inputs.squeeze(dim=0)
    imgs = [x for x in inputs]
    grid = make_grid(imgs).permute(1, 2, 0)
    plt.imshow(grid)
    plt.show()


if __name__ == "__main__":
    main()

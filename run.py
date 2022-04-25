import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
from dataset.base import PanAfDataset
from dataset.human import SupervisedPanAf


def main():

    print("=> Base class")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((244, 244))]
    )

    dataset = SupervisedPanAf(
        data_dir="/home/dl18206/Desktop/phd/data/panaf/obfuscated/videos",
        ann_dir="/home/dl18206/Desktop/phd/data/panaf/obfuscated/annotations/json_obfuscated",
        sequence_len=5,
        sample_itvl=1,
        transform=transform,
        behaviour_threshold=72,
    )

    print(dataset.__len__())

    dataloader = DataLoader(dataset)
    inputs, behaviour = next(iter(dataloader))

    print(f"Input {inputs.shape}, Label: {behaviour}")

    fig = plt.figure(figsize=plt.figaspect(0.75))
    inputs = inputs.squeeze(dim=0)
    imgs = [x for x in inputs]
    grid = make_grid(imgs).permute(1, 2, 0)
    plt.imshow(grid)
    plt.show()


if __name__ == "__main__":
    main()

# The Pan-African Dataset

The Pan-African dataset is the largest and most diverse open-access video dataset of great apes in the wild. The dataset comprises ∼20,000 videos and was curated from footage gathered at 18 study sites spanning several African countries. The dataset comprises frame-by-frame annotations for full-body location, intra-video ape identification (i.e. tracking data), species classification and basic behaviour/action. The video footage was originally collected by the Max Plank Institute (MPI) as part of the Pan African Programme, originally established to better understand the evolutionary-ecological drivers of behavioral diversity in great apes.

## What can be found here?

This is a PyTorch implemenation of the PanAfrican dataset.

<p align="center">
  <img src="https://user-images.githubusercontent.com/43569179/163564388-531c34f7-8ac0-4620-a2ff-3d2dc34fa324.jpg" width=400>
  <img src="https://user-images.githubusercontent.com/43569179/163564476-27c96484-c084-4247-b1ac-8652f294df50.jpg" width=400>
  <br>
  <img src="https://user-images.githubusercontent.com/43569179/163564509-48cec8eb-f7e5-49f4-a0ad-04ec473b0733.jpg" width=400>
  <img src="https://user-images.githubusercontent.com/43569179/163564522-a67f9c57-16f8-4c4a-b058-60fe174afa72.jpg" width=400>
</p>


## Installation

Install [Anaconda](https://docs.conda.io/en/latest/miniconda.html). Then create a conda environment using the environment file (conda-environment.yml) using the following command.

```bash
conda env create --name envname --file=conda-environment.yml
```

Activate this conda environment:

```bash
conda activate envname
```

This should install the requisite packages.


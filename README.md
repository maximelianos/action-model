# action-classifier

Predicting the action taken by the agent at time step t from image frames at times t-1, t, t+1.

## Setup environment

Create conda environment:

```
conda create -n torch python=3.8 numpy pyyaml setuptools cmake pandas pillow scikit-image scikit-learn tqdm jupyter notebook matplotlib seaborn conda-forge::screen conda-forge::tree
```

```
pip install torch==2.4.1 torchvision opencv-python timm torch-summary tensorflow==2.13.0
```

Install [Robomimic](https://robomimic.github.io/) repository. Copy the contents of this repo to the robomimic repo.

## Dataset

The datasets are in HDF5 files, making them easy to understand - see [dataset overview](https://robomimic.github.io/docs/datasets/overview.html). I downloaded the robomimic v0.1 dataset with the direct link (proficient human - lift real).

## Run

* `dataset_inspect.ipynb`: inspect the HDF5 file, extract images and robot actions.
* `train.ipynb`: main training notebook. Models used: ViT with head finetuning, small CNN from scratch.

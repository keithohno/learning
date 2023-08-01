## Setup

Install miniconda and set up an environment:

```
conda create -p env
conda activate ./env
conda install pytorch torchvision torchaudio -c pytorch
conda install matplotlib scikit-learn tqdm
```

## GPU support

This step is highly recommended if you have a GPU.

### Nvidia

Install the cuda drivers from the Nvidia website†, then add the pytorch-cuda package.

```
conda install pytorch-cuda -c nvidia
```

† I had issues with the .deb installation on newer versions of Ubuntu, but it worked on 18.04 LTS + WSL2

### Apple (MPS)

MPS support comes out of the box with the conda installation.

## Running examples

```
conda activate ./env
python main.py
```

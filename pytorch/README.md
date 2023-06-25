## Setup

Verified for Ubuntu 18.04 on WSL2. 

1. Install cuda drivers from Nvidia
    * Note: the deb installation didn't work for me on Ubuntu 22.04
2. Install Miniconda
3. Set up an environment:
```
conda create -p env
conda activate ./env
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
conda install jupyterlab matplotlib scikit-learn tqdm
```

## Running notebooks

```
conda activate ./env
jupyter lab
```

## Running examples
```
conda activate ./env
cd ${EXAMPLE_DIR}
python main.py
```

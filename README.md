# pytorch-learning

## What's this?

Julia + PyTorchを使ったディープラーニングの学習

***

## Setup

### Environment
- OS: Ubuntu 18.04 LTS
- Julia: `1.1.1`
- Python: `3.7.4` (Anaconda `4.5.11`)

### Installation
```bash
# --- Install python packages ---
# install jupyter notebook
$ conda install jupyter

# install PyTorch(CPU version)
$ conda install pytorch-cpu torchvision-cpu -c pytorch

# install Matplotlib
$ conda install -c conda-forge matplotlib


# --- Install julia packages ---
# install IJulia, PyCall
$ julia # julia REPL
julia> # press `]` key to enter package mode

## install IJulia: package for JupyterNotebook
pkg> add IJulia

## install PyCall: package for call python modules
pkg> add PyCall

# -> press `Ctrl + D` key to exit julia REPL
```

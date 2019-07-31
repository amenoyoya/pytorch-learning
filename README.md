# pytorch-learning

## What's this?

PyTorchを使ったディープラーニングの学習

***

## Setup

### Environment
- CPU: Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz
- GPU: NVIDIA Corporation GP104 `GeForce GTX 1070`
- Memory: 16GB
- OS: Ubuntu 18.04 LTS
- Python: `3.7.3` (Miniconda `4.6.14`)

### Installation
```bash
# --- Install Drivers ---
# auto install ubuntu drivers
$ sudo ubuntu-drivers autoinstall

# reboot
$ sudo reboot

# confirm nvidia driver
$ nvidia-smi

# install CUDA Toolkit 10.0
$ wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64 -O cuda-toolkit.deb
$ sudo dpkg -i cuda-toolkit.deb
$ sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
$ sudo apt update && sudo apt install -y cuda

# install CUDA Toolkit 10.0 patch
$ wget http://developer.download.nvidia.com/compute/cuda/10.0/Prod/patches/1/cuda-repo-ubuntu1804-10-0-local-nvjpeg-update-1_1.0-1_amd64.deb -O cuda-toolkit-patch.deb
$ dpkg -i cuda-toolkit-patch.deb

# download cuDNN Runtime and Development Libraries for CUDA 10.0
## from https://developer.nvidia.com/rdp/cudnn-download
## required: registration

# install cuDNN libraries
$ sudo dpkg -i libcudnn7_7.6.2.24-1+cuda10.0_amd64.deb
$ sudo dpkg -i libcudnn7-dev_7.6.2.24-1+cuda10.0_amd64.deb

# reboot
$ sudo reboot


# --- Install python (Miniconda) ---
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ chmod +x ./Miniconda3-latest-Linux-x86_64.sh
$ ./Miniconda3-latest-Linux-x86_64.sh


# --- Install python packages ---
# install jupyter notebook
$ conda install jupyter

# install pytorch + cuda 10.0
$ conda install pytorch cuda100 -c pytorch

# confirm
$ python -c 'import torch; print(torch.cuda.is_available())'
False
## 上手くGPUが認識されていない => 要確認

# 仕方ないのでCPU版PyTorchをインストール
$ conda install pytorch-cpu torchvision-cpu -c pytorch

# install Matplotlib
$ conda install -c conda-forge matplotlib
```

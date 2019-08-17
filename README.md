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

# install onnx (Open Neural Network Expression)
$ conda install onnx -c conda-forge

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

## install PyPlot: package for call matplotlib module
pkg> add PyPlot

## install JSON: package for JSON parser
pkg> add JSON

## install HTTP: package for HTTP messages
pkg> add HTTP

# -> press `Ctrl + D` key to exit julia REPL
```

### Setup in Google Colaboratory
マシンスペックが足りず、ローカル環境での開発が難しい場合は、[Google Colaboratory](https://colab.research.google.com) を使ってディープラーニングの開発ができる

#### Preparation
まず、Google Drive のマウントと Julua 1.1.1 のインストールを行う

- [colab/Julia_startup.ipynb](./colab/Julia_startup.ipynb) を [Google Drive](https://drive.google.com/drive/my-drive) にアップロードする
- **Julia_startup.ipynb** を Google Colaboratory で開く
    - 右クリック > アプリで開く > Google Colaboratory
- ランタイム > すべてのセルを実行 (`Ctrl + F9`)
    - Google Drive をマウント:
        - 認証用URLにアクセスし、アクセスを許可
        - 発行された認証トークンをコピーし、inputボックスに貼り付ける
        - マウントに成功すると `/content/drive/My Drive/` から Google Drive のデータにアクセスできるようになる
    - Julia 1.1.1 をインストール:
        - パッケージのダンロードとインストールが終わるまで待つ

#### Run Julia 1.1
上記の準備が完了したら、Julia 1.1 カーネルの Jupyter Notebook を実行する

- [colab/Julia1.1.ipynb](./colab/Julia1.1.ipynb) を [Google Drive](https://drive.google.com/drive/my-drive) にアップロードする
- **Julia1.1.ipynb** を Google Colaboratory で開く
    - 右クリック > アプリで開く > Google Colaboratory
- 動作確認用に以下のコードを実行
    ```julia
    for i = 1:3
        println("動作確認: $(i)回目")
    end
    ```
    - JuliaはJITコンパイルを行うため、初回起動時はそこそこ時間がかかる
    - 問題なく実行できたら成功

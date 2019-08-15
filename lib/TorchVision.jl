"""
Python PyTorch torchvision wrapper library

MIT License

Copyright (c) 2019 amenoyoya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

module TorchVision
    export transforms, models
    export predict
    export make_transformer
    export make_transformer_for_vgg16

    using PyCall

    const torch = pyimport("torch")
    const torchvision = pyimport("torchvision")
    const transforms = torchvision.transforms
    const models = torchvision.models

    # torch.Modelに予測させる
    ## Julia標準のArrayデータを入力として受け取り、Arrayで返す
    predict(model::PyObject, data::Array) = model(torch.tensor(data)).detach().numpy()

    # PIL.Imageオブジェクトに対して変換処理を行う関数を生成
    ## @param resize: transforms.Resizeオブジェクト
    ## @param crop: transforms.Cropオブジェクト
    ## @param normalize: transforms.Normalizeオブジェクト
    ## @return image::PyObject（PIL.Image） -> data::Array{AbstractFloat, 3}（色数×高さ×幅）
    make_transformer(resize::PyObject, crop::PyObject, normalize::PyObject) = begin
        transformer = transforms.Compose([
            resize
            crop
            transforms.ToTensor() # torch.Tensor形式に変換
            normalize
        ])
        return (image::PyObject) -> transformer(image).numpy()
    end

    # PIL.ImageオブジェクトをVGG16ニューラルネットワークモデル用のデータ形式に変換する関数を生成
    make_transformer_for_vgg16() = begin
        transformer = make_transformer(
            transforms.Resize(224), # 短い方の辺の長さが224になるようにリサイズ
            transforms.CenterCrop(224), # 画像中央を 224 x 224 で切り取り
            # 色平均化
            ## 色チャンネルの平均値: RGB = (0.485, 0.456, 0.406)
            ## 色チャンネルの標準偏差: RGB = (0.229, 0.224, 0.225)
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )
        # 値を0〜1の範囲内に収めて画像データ配列を返す関数を返す
        ## clamp(x, low, high): if x < low then x = low, if x > high then x = high
        ## `.`演算子: 関数を行列に対して適用させる
        return (image::PyObject) -> clamp.(transformer(image), 0, 1)
    end
end

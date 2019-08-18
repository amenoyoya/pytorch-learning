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
    export torch, torchvision, transforms, models
    export Color, Train, Valid, Phase
    export seed_random!, predict
    export make_transformer, make_transformer_for_vgg16, make_transformer_for_training, make_transformer_for_vgg16_training

    using PyCall, Random

    const torch = pyimport("torch")
    const torchvision = pyimport("torchvision")
    const transforms = torchvision.transforms
    const models = torchvision.models

    # 色表現型
    const Color = Tuple{AbstractFloat,AbstractFloat,AbstractFloat}

    # 機械学習モード表現型
    abstract type Train end
    abstract type Valid end
    const Phase = Union{Type{Train}, Type{Valid}}

    # 型（CamelCase）を文字列（snake_case）に変換する
    macro typestr(T)
        replace(replace(string(T), r"([A-Z])" => c -> "_" * lowercase(c)), r"^_" => s"")
    end

    # 乱数初期化
    ## 値を破壊的に変更する（もしくはグローバル変数の状態に影響を与える）関数には慣例的に`!`をつける
    seed_random!(seed::Int) = begin
        Random.seed!(seed)
        torch.manual_seed(seed) # PyTorchの乱数初期化
    end

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
    # () -> ((img::PyObject) -> Array{Float32,3})
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

    # モデル訓練用に入力画像を変換する関数を生成
    # (resize, std, mean) -> ((img::PyObject, phase::Phase) -> Array{Float32,3})
    ## @param resize::Int = リサイズする大きさ
    ## @param std::Color = 各色チャンネルの平均値
    ## @param mean::Color = 各色チャンネルの標準偏差
    make_transformer_for_training(resize::Int, std::Color, mean::Color) = begin
        trasformers = Dict(
            "train" => make_transformer(
                transforms.RandomResizedCrop(resize; scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean, std)
            ),
            "valid" => make_transformer(
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.Normalize(mean, std)
            )
        )
        return (img::PyObject, phase::Phase) -> trasformers[@typestr phase](img)
    end

    # VGG-16モデル訓練用に入力画像を変換する関数を生成
    # (resize, std, mean) -> ((img::PyObject, phase::Phase) -> Array{Float32,3})
    make_transformer_for_vgg16_training() = begin
        transformer = make_transformer_for_training(224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return (img::PyObject, phase::Phase) -> clamp.(transformer(img, phase), 0, 1)
    end

    # PyTorch規格の画像データセット型生成マクロ
    # 要 PyCall, ./Image
    # @param TypeName: 型名
    # @param make_dataset_list_function(phase::Phase)::Array{String,1}: データセットのファイルパスリストを返す関数
    # @param image_transform_function(img::PyObject, phase::Phase)::Array{Float32,3}: 画像をモデル入力用に変換する関数
    # @param labeling_function(img_path::String)::Int: データセットのラベルインデックスを返す関数
    ## esc(SymbolExpression) を使うことで マクロ実行位置にそのままコードを埋め込むことができる
    ## ※ そのまま埋め込まれるため変数スコープ等は効かない
    macro image_dataset(TypeName, make_dataset_list_function, image_transform_function, labeling_function)
        esc(quote
            @pydef mutable struct $TypeName <: TorchVision.torch.utils.data.Dataset
                __init__(self, phase::TorchVision.Phase) = begin
                    pybuiltin(:super)($TypeName, self).__init__()
                    self.phase = phase
                    self.file_list = $make_dataset_list_function(phase)
                end
                
                __len__(self) = length(self.file_list)
                
                __getitem__(self, index::Int) = begin
                    # index番目の画像をロード
                    ## Juliaのindexは1〜なので +1 する
                    img_path = self.file_list[index + 1]
                    img = Image.open(img_path).convert("RGB") # グレースケール画像は強制的にRGB画像に変換
                    img_transformed = $image_transform_function(img, self.phase)
                    # ラベリング
                    label = $labeling_function(img_path)
                    return img_transformed, label
                end
            end
        end)
    end
end

module PyImage
    using PyCall, PyPlot

    const PIL = pyimport("PIL")
    const torchvision = pyimport("torchvision")
    const plt = PyPlot.matplotlib.pyplot
    const io = pyimport("io")

    export transforms
    const transforms = torchvision.transforms

    # 画像をファイルから読み込み
    open(filepath::String)::PyObject = PIL.Image.open(filepath)

    # bytesから画像生成
    ## byte: UInt8型1次元配列
    open(bytes::Array{UInt8, 1})::PyObject = PIL.Image.open(io.BytesIO(bytes))

    # PIL.Imageオブジェクトに対して変換処理を行う関数を生成
    ## @param compose: PIL.transformsの配列（変換方法の指定）
    ## @return image::PyObject（PIL.Image） -> data::Array{AbstractFloat, 3}（色数×高さ×幅）
    make_transformer(compose::Array{PyObject, 1})::(PyObject -> Array{AbstractFloat, 3}) = begin
        transformer = transforms.Compose(hcat(compose, [transforms.ToTensor()]))
        (image::PyObject) -> transformer(image).numpy()
    end

    # 画像表示
    show(image::Any)::PyObject = plt.imshow(image)
end

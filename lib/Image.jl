"""
Python PIL wrapper library

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

module Image
    using PyCall, PyPlot

    const PIL = pyimport("PIL")
    const plt = PyPlot.matplotlib.pyplot
    const io = pyimport("io")

    # 画像をファイルから読み込み
    open(filepath::String)::PyObject = PIL.Image.open(filepath)

    # bytesから画像生成
    ## byte: UInt8型1次元配列
    open(bytes::Array{UInt8, 1})::PyObject = PIL.Image.open(io.BytesIO(bytes))

    # 画像表示
    ## PIL.Imageオブジェクト or Array{Float32, 3}(高さ×幅×色数)形式のデータを表示
    show(image::Any)::PyObject = plt.imshow(image)

    # Matplotlibビューワー起動
    ## Jupyter Notebook ではインライン表示されるため不要
    viewer()::PyObjet = plt.show()
end

"""
ImageNetから画像ダウンロード
@purpose: ハリネズミとヤマアラシを識別するためのデータセットを収集
"""

using HTTP

# ダウンロード先のディレクトリ作成
if !isdir("dataset")
    mkdir("dataset")
end

# 画像ダウンロード関数: (url::String, savedir::String) -> filesize::Int
download_image(url::String, savedir::String="dataset")::Int = begin
    # ダウンロード済みならスキップ
    if isfile("$(savedir)/$(basename(url))")
        return 0
    end

    try
        # タイムアウト=30秒, 再試行なし
        r = HTTP.request("GET", url; readtimeout=30, retry=false)
        return open("$(savedir)/$(basename(url))", "w") do fp
            write(fp, r.body)
        end
    catch
        return 0
    end
end

# ImageNetのヤマアラシ・ハリネズミ画像をダウンロード
## URLリストを読み込み、各URLから画像を取得
r = HTTP.request("GET", "http://imagenet.stanford.edu/api/text/imagenet.synset.geturls?wnid=n02346627")
lines = split(String(r.body), r"[\n\r]") # 改行で文字列分解
for url in lines[lines .!= ""] # 空文字でない行について順次処理
    println("$(url): downloaded size = $(download_image(url))")
end

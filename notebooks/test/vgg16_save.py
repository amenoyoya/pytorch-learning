'''
PyTorch: 学習済みVGG-16モデルの保存
'''
import torch
from torchvision import models
from torch.autograd import Variable
import torch.onnx

# 学習済みVGG-16モデルロード
vgg16 = models.vgg16(pretrained=True)

# 学習済みモデルの保存（パラメータのみ）
# torch.save(vgg16.state_dict(), './vgg16_weight.pth')

# ONNX形式でモデル保存
## ONNX形式ならJulia＋Fluxでロードできるはず。。。
# dummy_inputは、モデルの入力テンソルに合わせる
## vgg16は tensor.Size(1, 3, 224, 224) の入力を受け付けるため、その形式のランダムなテンソルを作成して渡す
dummy_input = Variable(torch.randn(1, 3, 224, 224))
torch.onnx.export(vgg16, dummy_input, 'vgg16.onnx', verbose=True)

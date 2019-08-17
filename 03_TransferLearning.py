'''
PyTorch: 転移学習
'''

import os, glob, random, json, torch, torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import models, transforms

# 乱数シーディング
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

def make_transformer(resize, mean, std):
    transformer = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
    def transform(img, phase='train'):
        return np.clip(transformer[phase](img).numpy(), 0, 1)
    return transform

transform = make_transformer(224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def make_dataset_list(phase='train'):
    return glob.glob(f'./dataset/{phase}/**/*.jpg')

# ハリネズミとヤマアラシのデータセット作成
class Dataset(data.Dataset):
    def __init__(self, phase='train'):
        super(Dataset, self).__init__()
        self.phase = phase
        self.file_list = make_dataset_list(phase)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        # index番目の画像をロード＆変換
        path = self.file_list[index]
        img = Image.open(path).convert('RGB') # グレースケールの画像は入力できないためRGB画像に変換してロード
        img_transformed = transform(img, self.phase)
        # 画像のラベル名をパスから抜き出す
        label = path[len(self.phase) + 11 : len(self.phase) + 19]
        label = (0 if label == 'hedgehog' else 1)
        return img_transformed, label

train_dataset = Dataset('train')
val_dataset = Dataset('val')

# ミニバッチサイズ
batch_size = 32

# DataLoader作成
train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_dataloader = data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True
)
dataloaders = {
    'train': train_dataloader,
    'val': val_dataloader
}

# 学習済みVGG-16モデルのロード
net = models.vgg16(pretrained=True)

# VGG-16の最後の出力層の出力ユニットを2個に付け替える
net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

# 訓練モードに設定
net.train()

# 損失関数の定義
criterion = nn.CrossEntropyLoss()

# 転移学習で学習させるパラメータを params_to_update に格納
params_to_update = []

# 学習させるパラメータ名
update_param_names = ["classifier.6.weight", "classifier.6.bias"]

# 学習させるパラメータ以外は勾配計算させない
for name, param in net.named_parameters():
    if name in update_param_names:
        param.required_grad = True
        params_to_update += [param]
        print(name)
    else:
        param.required_grad = False

# 最適化手法の設定
optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)

# モデル訓練
def train_model(net, dataloaders, criterion, optimizer, num_epochs):
    # epoch数分ループ
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}")
        print("----------")
        
        # epochごとの学習と検証のループ
        for phase in ["train", "val"]:
            if phase == "train":
                net.train() # 訓練モードに
            else:
                net.eval() # 検証モードに
            
            epoch_loss = 0.0 # epochの損失和
            epoch_corrects = 0 # epochの正解数
            
            # 未学習時の検証性能を確かめるため、最初の訓練は省略
            if epoch == 1 and phase == "train":
                continue
            
            # データローダーからミニバッチを取り出すループ
            for inputs, labels in tqdm(dataloaders[phase]):
                # optimizer初期化
                optimizer.zero_grad()
                
                # 順伝搬計算
                torch.set_grad_enabled(phase == "train")
                outputs = net(inputs)
                loss = criterion(outputs, labels) # 損失計算
                _, preds = torch.max(outputs, 1) # ラベルを予測
                # 訓練時はバックプロパゲーション
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                # イテレーション結果の計算
                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data)
                torch.set_grad_enabled(False)
            
            # epochごとの損失と正解率を表示
            epoch_loss = epoch_loss / len(dataloaders[phase].dataset)
            epoch_acc = epoch_corrects ** 2 / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss}, Acc: {epoch_acc}")

# 学習・検証を実行
train_model(net, dataloaders, criterion, optimizer, 2)

# 転移学習結果の確認
net.eval() # 推論モードに設定
inputs = [
    transform(Image.open("./data/gahag-0059907781-1.jpg")),
    transform(Image.open("./data/publicdomainq-0025120muq.jpg")),
]
pred = net(torch.Tensor(inputs))
print(pred)

'''
-> 結果
[
    ハリネズミ, ヤマアラシ
    [   0.2958,  0.2882],
    [  -0.0021,  0.2396]
]
'''
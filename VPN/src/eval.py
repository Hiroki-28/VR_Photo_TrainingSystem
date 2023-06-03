import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import transforms
import os, sys, glob
import numpy as np
import pandas as pd
from PIL import Image
import warnings
from net import SSD_320 as ssd_320
import progressbar
warnings.simplefilter('ignore', UserWarning)


#データセット作成 (ペア画像を返す)
class PairImages(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform):
        img_paths = img_dir  #ペア画像を入れたフォルダ(数十万)の親パスを指定
        self.imgs_list = glob.glob(os.path.join(self.img_paths, "*"))  #ペア画像を入れたフォルダ名のリスト取得
        self.transform = transform
    
    def __getitem__(self, index):
        #画像を読み込む
        composition_A = Image.open(os.path.join(self.imgs_list[index], "composition_A.jpg"))
        composition_B = Image.open(os.path.join(self.imgs_list[index], "composition_B.jpg"))
	
	#画像の正規化
        composition_A = self.transform(composition_A)
        composition_B = self.transform(composition_B)
        return composition_A, composition_B  #構図のスコアは、compositionA > compositionB (ペア画像を作成する段階で、構図が良い方をA、悪い方をBとしている)
       
    def __len__(self):
        return len(self.imgs_list)
        #return len(glob.glob(self.img_paths + "*"))


class MPSELoss(nn.Module):
    def __init__(self):
        super(MPSELoss, self).__init__()

    def forward(self, outputs, targets):
        count = 0
        for idx, (output, target) in enumerate(zip(outputs, targets)):
            for i in range(len(output)):
                for j in range(i+1, len(output)):
                    if idx == 0 and i == 0 and j == 1:
                        loss = ((output[i]-output[j]) - (target[i]-target[j])) ** 2
                    else:
                        loss += ((output[i]-output[j]) - (target[i]-target[j])) ** 2
                    count += 1
        loss = loss / count
        return loss


def valid_transform(image_size=224):
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        #transforms.Normalize(mean=[0.4086, 0.4000, 0.3717], std=[0.2285, 0.2072, 0.2035])])
    return valid_transform


def main():
    #parserを作成
    parser = argparse.ArgumentParser(description="Full VGG trained on CPC")
    parser.add_argument('--l1', default=1024, type=int)
    parser.add_argument('--l2', default=512, type=int)
    parser.add_argument('--resume', '-r', default='./model_pth/CPC/VPN_20221202.pth', type=str, help='resume from checkpoint')
    args = parser.parse_args()

    #データセット作成 (テストデータを読み込む)
    test_dataset = PairImages("./dataset/XPViewDataset/Pair_images_95", transform=valid_transform(320))
    #データセットをデータローダーに読み込ませる
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)
    print('len(test_loader):', len(test_loader))
    print('len(test_loader.dataset):', len(test_loader.dataset))

    #VPNモデルを読み込む
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    VPN_net = ssd_320().to(device)
    criterion = MPSELoss()
    #学習済みモデルのパラメータを読み込む
    VPN_net.load_state_dict(torch.load(args.resume))

    #評価モード
    VPN_net.eval()
    swap_error_sum = 0
    test_loss = 0
    
    #評価では、勾配の計算はしない
    with torch.no_grad():
        for compA, compB in test_loader:
            compA, compB = Variable(compA.to(device)), Variable(compB.to(device))
            #順伝播を計算
            output1, output2 = VPN_net(compA), VPN_net(compB)
            #swap error(SW)の計算
            for oup1, oup2 in zip(output1[0], output2[0]):
                if oup1.item() < oup2.item():
                    swap_error_sum += 1
    
    #swap error(SW)の平均を計算
    swap_error = swap_error_sum / len(test_loader.dataset)
    print('swap_error:{0:.3f}'.format(swap_error))


if __name__ == '__main__':
    main()


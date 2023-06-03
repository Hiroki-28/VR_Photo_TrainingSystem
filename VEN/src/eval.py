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
from nets.FullVggCompositionNet import FullVggCompositionNet as CompositionNet
from nets.SiameseNet import SiameseNet
import progressbar
warnings.simplefilter('ignore', UserWarning)


#データセット作成（ペア画像を返す）
class PairImages(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform):
        self.img_paths = img_dir  #画像ペアを入れたフォルダ(数十万)の親パスを指定
        self.imgs_list = glob.glob(os.path.join(self.img_paths, "*"))  #画像ペアを入れたフォルダ名のリストを取得
        self.transform = transform
    
    def __getitem__(self, index):
        #画像を読み込む
        composition_A = Image.open(os.path.join(self.imgs_list[index], "composition_A.jpg"))
        composition_B = Image.open(os.path.join(self.imgs_list[index], "composition_B.jpg"))
	
	#画像の正規化
        composition_A = self.transform(composition_A)
        composition_B = self.transform(composition_B)
        return composition_A, composition_B  #構図のスコアは、compositionA > compositionBである。 (ペア画像を作成する段階で、構図が良い方をA、悪い方をBとしている)
       
    def __len__(self):
        return len(self.imgs_list)  #画像ペアのフォルダ数 = データセット数


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
    parser.add_argument('--resume', '-r', default='./model_pth/VRPhoto/VEN_20221206.pth', type=str, help='resume from checkpoint')
    args = parser.parse_args()

    #データセット作成 (テストセットを読み込む)
    test_dataset = PairImages("./datasets/XPViewDataset/Pair_images_95", transform=valid_transform(224))  #インスタンス生成
    print("test_dataset:", test_dataset)
    #データセットをデータローダーに読み込ませる
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)
    print('len(test_loader):', len(test_loader))
    print('len(test_loader.dataset):', len(test_loader.dataset))

    #VGG16とsiameseのモデルを読み込む
    device = "cuda" if torch.cuda.is_available() else "cpu"
    single_pass_net = CompositionNet(pretrained=False, LinearSize1=args.l1, LinearSize2=args.l2).to(device)
    siamese_net = SiameseNet(single_pass_net).to(device)
    
    #学習済みモデルのパラメータを読み込む
    ckpt_file = args.resume
    siamese_net.load_state_dict(torch.load(ckpt_file))
    #ckpt = torch.load(ckpt_file, map_location=lambda storage, loc: storage)  #ckptは辞書型。重み・バイアスを保存しているのは、"state_dictキー"
    #model_state_dict = ckpt['state_dict']
    #siamese_net.load_state_dict(model_state_dict)

    #損失関数
    criterion = nn.MarginRankingLoss(margin=1)

    #評価モード
    siamese_net.eval()
    swap_error_sum = 0
    test_loss = 0
    
    #評価では、勾配の計算はしない
    with torch.no_grad():
        for compA, compB in test_loader:
            compA, compB = Variable(compA.to(device)), Variable(compB.to(device))
            #順伝播を計算
            output1, output2 = siamese_net(compA, compB)
	    #swap error(SW)の計算
            for oup1, oup2 in zip(output1, output2):
                #print("oup1:{0}, oup2:{1}".format(oup1.item(), oup2.item()))
                if oup1.item() < oup2.item():
                    swap_error_sum += 1
	    #1バッチあたりの平均lossを計算 (バッチごと)
            loss = criterion(output1, output2, torch.ones_like(output1))
            test_loss += loss.item()
            #print("test_loss:{0:.6f}".format(loss.item()))
    
    print("ckpt_file:", ckpt_file)
    #swap error(SW)の平均を計算
    swap_error = swap_error_sum / len(test_loader.dataset)
    print('swap_error:{0:.6f}'.format(swap_error))
    #1エポックあたりの平均lossを計算 (エポックごと)
    avg_test_loss = test_loss / len(test_loader)
    print('test_loss: {0:.6f}'.format(avg_test_loss))


if __name__ == '__main__':
    main()

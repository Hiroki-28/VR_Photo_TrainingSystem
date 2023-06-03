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


#データセット作成（indexの値以下の数の画像ペアを返す）
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
        return len(self.imgs_list)  #画像ペアのフォルダ数 = データ数


class EarlyStopping:
    def __init__(self, patience, path):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
    
    def __call__(self, model, valid_loss):
        if self.best_loss < valid_loss:
            self.counter += 1
            print("EarlyStopping counter: {0}/{1}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.checkpoint(model, valid_loss)
            self.counter = 0

    def checkpoint(self, model, valid_loss):
        print("Validation loss decreased ({0:.4f} ==> {1:.4f})".format(self.best_loss, valid_loss))
        torch.save(model.state_dict(), self.path)
        self.best_loss = valid_loss


def get_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    return directory


def train_transform(image_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        #transforms.Normalize(mean=[0.4086, 0.4000, 0.3717], std=[0.2285, 0.2072, 0.2035])])
    return train_transform


def valid_transform(image_size=224):
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        #transforms.Normalize(mean=[0.4086, 0.4000, 0.3717], std=[0.2285, 0.2072, 0.2035])])
    return valid_transform


def main():
    #parser
    parser = argparse.ArgumentParser(description="Full VGG trained on CPC")
    parser.add_argument('--dataset_dir', default='./datasets/VRPhoto/PairImages_All_5_1', type=str, help='ペア画像を入れたフォルダの親パス')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=150, type=int)
    parser.add_argument('--learning_rate', default=2e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--step_size', default=30, type=int, help='学習率を変更する周期')
    parser.add_argument('--gamma', default=0.5, type=float, help='学習率を変更する倍率')
    parser.add_argument('--param_savedir', default='./model_pth/VRPhoto', type=str)
    parser.add_argument('--csvfile_savedir', default='./output/VRPhoto', type=str)
    parser.add_argument('--date', default='20221216', type=str)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--resume', default='./config/EvaluationNet.pth.tar', type=str)
    args = parser.parse_args()

    #フォルダ作成・ハイパーパラメータ保存
    param_savedir = get_dir(args.param_savedir)
    csvfile_savedir = get_dir(args.csvfile_savedir)
    df = pd.DataFrame({'dataset_dir': [args.dataset_dir], 'batch_size': [args.batch_size], 'learning_rate': [args.learning_rate], 'momentum': [args.momentum], 'weight_decay':[args.weight_decay], 'step_size': [args.step_size], 'gamma': [args.gamma]}).T
    df.to_csv(f"{csvfile_savedir}/VEN_config_{args.date}.csv", header=None)

    #データセット作成 (trainingデータとvalidationデータを9：1に分割)
    dataset = PairImages(img_dir=args.dataset_dir, transform=train_transform(224))  #インスタンス生成
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset) - int(len(dataset)*0.9)], generator=torch.Generator().manual_seed(0))  #乱数固定
    #作成したデータセットをデータローダーに読み込ませる
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    #モデルの定義（モデルの読み込み）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    single_pass_net = CompositionNet(pretrained=True, LinearSize1=1024, LinearSize2=512).to(device)
    siamese_net = SiameseNet(single_pass_net).to(device)
    #single_pass_net = CompositionNet().to(device)
    criterion = nn.MarginRankingLoss(margin=1)  # loss function
    optimizer = torch.optim.SGD(siamese_net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)  # optimization function
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) # learning schedule instance
    earlystopping = EarlyStopping(patience=args.patience, path=os.path.join(param_savedir, f'VEN_{args.date}.pth'))

    #学習済みモデルのパラメータを読み込む
    ckpt_file = args.resume
    ##siamese_net.load_state_dict(torch.load(ckpt_file))
    ckpt = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
    model_state_dict = ckpt['state_dict']
    siamese_net.load_state_dict(model_state_dict)

    #学習
    num_epochs = args.num_epochs
    train_loss_list = []
    valid_loss_list = []
    train_swaperror_list = []
    valid_swaperror_list = []
    epoch_list = []
    print("len(train_loader):", len(train_loader))
    print("len(valid_loader):", len(valid_loader))
    for epoch in range(num_epochs):
        print("{}epoch".format(epoch+1))
        #エポックごとに初期化
        train_loss = 0
        valid_loss = 0
        sum_train_swaperror = 0
        sum_valid_swaperror = 0

        #=====train=====
        #訓練モードに切り替え
        siamese_net.train()
        batch_count = 0
        #バッチサイズごとに読み込む(64個ずつのはず)
        for compA, compB in train_loader:
            compA, compB = compA.to(device), compB.to(device)
            #勾配を初期化
            optimizer.zero_grad()
            #順伝播の計算
            output1, output2 = siamese_net(compA, compB) #xは入力画像。出力2つ(output1, output2)
            #swap errorの計算
            for oup1, oup2 in zip(output1, output2):
                if oup1.item() < oup2.item():
                    sum_train_swaperror += 1
            #1バッチあたりの平均lossを計算 (バッチごと)
            loss = criterion(output1, output2, torch.ones_like(output1)) #損失は、バッチ内の損失を平均化したもの
            train_loss += loss.item()
            #逆伝播の計算
            loss.backward()
            optimizer.step()
            #batch countを表示
            batch_count += 1
            if batch_count % 200 == 0:
                print("count:{0}, loss:{1}".format(batch_count, loss))
	    #swap errorの平均を計算
        avg_train_swaperror = sum_train_swaperror / len(train_loader.dataset)
        #1エポックあたりの平均lossを計算
        avg_train_loss = train_loss / len(train_loader)

        #=====valid=====
        #評価モードに切り替え
        siamese_net.eval()
        #評価では、勾配の計算はしない。メモリ消費量の削減
        with torch.no_grad():
            for compA, compB in valid_loader:
                compA, compB = compA.to(device), compB.to(device)
                #順伝播を計算
                output1, output2 = siamese_net(compA, compB)
                #swap errorの計算
                for oup1, oup2 in zip(output1, output2):
                    if oup1.item() < oup2.item():
                        sum_valid_swaperror += 1
                #1バッチあたりの平均lossを計算 (バッチごと)
                loss = criterion(output1, output2, torch.ones_like(output1))
                valid_loss += loss.item()
        #swap errorの平均を計算
        avg_valid_swaperror = sum_valid_swaperror / len(valid_loader.dataset)
        #1エポックあたりの平均lossを計算 (エポックごと)
        avg_valid_loss = valid_loss / len(valid_loader)
        print('Epoch [{0}/{1}], loss: {2:.4f}, val_loss: {3:.4f}'.format(epoch+1, num_epochs, avg_train_loss, avg_valid_loss))
        
        #loss,swaperrorをリストに追加
        epoch_list.append(epoch+1)
        train_loss_list.append(avg_train_loss)
        valid_loss_list.append(avg_valid_loss)
        train_swaperror_list.append(avg_train_swaperror)
        valid_swaperror_list.append(avg_valid_swaperror)

        #lossの推移を出力
        df = pd.DataFrame({'epoch': epoch_list, 'loss': train_loss_list, 'val_loss': valid_loss_list})
        df2 = pd.DataFrame({'epoch': epoch_list, 'swaperror': train_swaperror_list, 'val_swaperror': valid_swaperror_list})
        df.to_csv(f"{csvfile_savedir}/VEN_loss_{args.date}.csv", index=None)
        df2.to_csv(f"{csvfile_savedir}/VEN_swaperror_{args.date}.csv", index=None)

        #earlystoppingのcall関数を呼び出し
        earlystopping(siamese_net, avg_valid_loss)
        if earlystopping.early_stop:
            print("Early Stopping!")
            break

        #学習率を減衰させる
        scheduler.step()


if __name__ == '__main__':
    main()

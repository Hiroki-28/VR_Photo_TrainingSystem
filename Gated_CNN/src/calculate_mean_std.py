import torch
from torchvision import transforms, datasets
import os, sys, glob
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True

#カスタムデータセットの読み込み
class LoadImages(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir  #画像ファイルを入れたフォルダを指定
        self.imgs_list = glob.glob(os.path.join(self.img_dir, "*", "*.jpg"))  #画像ファイルのリストを取得
        self.transform = transform
    
    def __getitem__(self, index):
        #画像を読み込む
        img = Image.open(self.imgs_list[index])
        img = self.transform(img)
        return img
       
    def __len__(self):
        return len(self.imgs_list)  #画像のデータ数

    
def get_transform(image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    return transform


if __name__ == '__main__':

    #画像サイズ
    image_size = 224

    #データセットを読み込む
    #dataset = LoadImages("../dataset/AVA/CHE_portraiture_dataset/train", transform=get_transform(image_size))  #インスタンス生成
    dataset = datasets.ImageFolder(root="../dataset/VRPhoto/all", transform=get_transform(224))
    data_size = len(dataset)
    print("data_size:", data_size)

    #作成したデータセットをデータローダーに読み込ませる
    image_loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=0)

    #ピクセル値の和、ピクセル値の2乗和(チャネルごとに求める)
    pixel_sum = torch.tensor([0.0, 0.0, 0.0])
    pixel_square_sum = torch.tensor([0.0, 0.0, 0.0])
    
    for inputs, labels in image_loader:
    #for inputs in image_loader:
        #print("inputs.size", inputs.size())
        #sys.exit("end")
        pixel_sum += inputs.sum(axis=[0, 2, 3])  #バッチに含まれるすべてのサンプルの和を求めている。inputs.shape -> 64, 3, 224, 224
        pixel_square_sum += (inputs ** 2).sum(axis=[0, 2, 3])

    #ピクセル数の合計
    count = data_size * image_size * image_size

    #チャネルごとの平均と標準偏差を求める
    total_mean = pixel_sum / count
    total_var  = (pixel_square_sum / count) - (total_mean ** 2) #2乗の平均 - 平均の2乗
    total_std  = torch.sqrt(total_var)
    
    #meanとstdを出力
    print('mean:', total_mean)
    print('std:', total_std)
    #df = pd.DataFrame({'color': ['R', 'G', 'B'], 'mean': [total_mean[i].item() for i in range(3)], 'std': [total_std[i].item() for i in range(3)]})
    #df.to_csv("mean_std.csv", index=None)


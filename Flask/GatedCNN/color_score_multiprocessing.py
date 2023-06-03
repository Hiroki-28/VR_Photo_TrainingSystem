import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import transforms, datasets
import os, sys, glob
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import time
from net import ResNet, ResNet_GatedCNN, ColorHarmonyNet
from flask import Flask, request
from io import BytesIO
import multiprocessing
from multiprocessing import Process, Manager


# modelの読み込みを冒頭で行う。MultiProcessingではグローバル変数を読み込めない(if __name__ == "__main__"関数内で定義してもダメ)ために、model情報だけはあらかじめ定義する
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
color_net = ResNet_GatedCNN().to(device)
color_harmony_net = ColorHarmonyNet(color_net).to(device)
color_harmony_net.load_state_dict(torch.load("./model_pth/VRPhoto/CHE_20221204.pth", map_location=torch.device("cpu")))
color_harmony_net.eval()


def val_transform(image_size=224):
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3876, 0.3853, 0.3558], std=[0.2125, 0.1957, 0.1948])])
    return valid_transform


def get_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path


def calc_probability(patche):
    # calcurate sum of the probability
    prob = color_harmony_net(patche[0], patche[1], patche[2], patche[3], patche[4]).data.cpu().numpy()[0][0]
    return prob


app = Flask(__name__)
@app.route('/sample', methods=['POST'])
def main():
    # 画像を取得
    img = Image.open(BytesIO(request.get_data().split(b'\r\n--')[1].split(b'\r\n\r\n')[1])).convert('RGB')
    # img = Image.open("img_1.jpg")
    # 時間計測開始
    start = time.time()
    
    # transform変換
    img = transform(img).to(device).unsqueeze(0)  # shape: (3, 224, 224) -> (1, 3, 224, 224)
    ### calculate the probability of being good aesthetic
    patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute([0, 2, 3, 1, 4, 5])

    patches_list = []
    for i in range(1, patch_num_edge-1):
        for j in range(1, patch_num_edge-1):
            patches_center = patches[:, i, j, :, :, :]       # center, [32, 3, 16, 16]
            patches_neighbor1 = patches[:, i-1, j, :, :, :]  # upper , [32, 3, 16, 16]
            patches_neighbor2 = patches[:, i, j-1, :, :, :]  # left  , [32, 3, 16, 16]
            patches_neighbor3 = patches[:, i, j+1, :, :, :]  # right , [32, 3, 16, 16]
            patches_neighbor4 = patches[:, i+1, j, :, :, :]  # lower , [32, 3, 16, 16]
            patches_list.append([patches_center, patches_neighbor1, patches_neighbor2, patches_neighbor3, patches_neighbor4])

    with multiprocessing.Pool(processes=3) as p:
        scores_list = list(p.map(calc_probability, patches_list))
    score = sum(scores_list)/14
    if (score > 10):
        score = 10.000
    score = round(score, 1)
    
    # 時間計測終了
    end = time.time()
    print("\ndiff:", end - start)
    print("score:", score)

    response = str(score)
    return response


if __name__ == "__main__":
    # load_model("./model_pth/VRPhoto/CHE_20221204.pth")
    transform = val_transform(224)
    patch_size = 16
    patch_num_edge = int(224 / patch_size)
    print("------- starting server -------")
    app.run(host='0.0.0.0', port=5003)
    # main()


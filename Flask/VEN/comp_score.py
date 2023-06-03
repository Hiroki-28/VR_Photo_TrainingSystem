import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import transforms
import os, sys, csv
import numpy as np
import pandas as pd
from PIL import Image
import time
import json
from nets.FullVggCompositionNet import FullVggCompositionNet as CompositionNet
from nets.SiameseNet import SiameseNet
from py_utils import bboxes
from flask import Flask, request
from io import BytesIO


def get_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path


def val_transform(image_size=224):
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return valid_transform


def load_model(model_path):
    global device, single_pass_net
    # model define
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    single_pass_net = CompositionNet(pretrained=False, LinearSize1=1024, LinearSize2=512).to(device)
    siamese_net = SiameseNet(single_pass_net)
    siamese_net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    single_pass_net.eval()


app = Flask(__name__)
@app.route('/sample', methods=['POST'])
def main():
    img = Image.open(BytesIO(request.get_data().split(b'\r\n--')[1].split(b'\r\n\r\n')[1])).convert('RGB')
    start = time.time()
    # transform変換
    img = transform(img).to(device)
    # モデル入力
    inp = Variable(img).unsqueeze(0)   # shape: (3, 224, 224) -> (1, 3, 224, 224)
    # モデル出力
    oup = single_pass_net(inp)
    # スコア取り出し
    score = oup.data.cpu().numpy()[0][0]
    # scoreのほとんどが-10 < score < 10を満たす。ここでは、これを0 < score < 10に変換する (0点以上10点以下のスコアに変換)
    score = (score + 10) / 2
    if score > 10:
        score = 10.0
    elif score < 0:
        score = 0.0
    score = round(score, 1)
    end = time.time()
    print("\ndiff:", end - start)
    print("score:", score)
    response = str(score)

    return response



if __name__ == '__main__':
    load_model("./model_pth/VRPhoto/VEN_20221206.pth")
    transform = val_transform(224)
    print("------- starting server -------")
    app.run(host='0.0.0.0', port=5002)
    # main()


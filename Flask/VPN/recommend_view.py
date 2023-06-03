import os
import glob
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import pandas as pd
import numpy as np
import sys
# sys.path.append("../..")
from net import SSD_320 as ssd_320
from flask import Flask, request, jsonify
from io import BytesIO
import time
import base64


def GetImageCrops(image_path, pdefined_anchors):
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        print("The type of image_path is not str")
    
    w, h = image.size
    img_crops = []
    img_bboxes = []
    for anchor in pdefined_anchors:
        x1, y1, x2, y2 = int(anchor[0]*w), int(anchor[1]*h), int(anchor[2]*w), int(anchor[3]*h)
        img_crop = image.crop((x1,y1,x2,y2)).copy()
        img_crops.append(img_crop)
        img_bboxes.append([x1,y1,x2,y2])
    return img_crops, img_bboxes, w, h


def get_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    return directory


def bboxes_jaccard(bboxes1, bboxes2):
    """Computing jaccard index (IOU) between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    if isinstance(bboxes1, (tuple, list)):
        bboxes1 = np.array(bboxes1)
    if isinstance(bboxes2, (tuple, list)):
        bboxes2 = np.array(bboxes2)

    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    jaccard = int_vol / (vol1 + vol2 - int_vol)
    return jaccard


def bboxes_overlap_check(scores, bboxes, nms_threshold=0.45):
    if isinstance(scores, list):
        scores = np.array(scores)
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)

    keep_bboxes = np.ones(len(scores), dtype=np.bool_)
    for i in range(len(scores)-1):
        if keep_bboxes[i]:
            overlap = bboxes_jaccard(bboxes[i], bboxes[(i+1):])
            keep_overlap = (overlap < nms_threshold)
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)
    idxes = np.where(keep_bboxes)
    return scores[idxes].tolist(), bboxes[idxes].tolist(), idxes


def val_transform(image_size=320):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform


def load_model(model_path):
    global device, VPN_net
    # model define
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    VPN_net = ssd_320(num_classes=500).to(device)
    # VPN_net.load_state_dict(torch.load(args.model_path))
    VPN_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    VPN_net.eval()


def predict(image):
    w, h = image.size
    inp = transform(image).to(device)
    oup = VPN_net(inp.unsqueeze(0))
    oup_numpy = oup.squeeze(0).data.cpu().numpy()
    scores_list = [oup_numpy[i] for i in range(len(oup_numpy))]  # len(oup_numpy) -> default:500
    bboxes_list = [[int(anchor[0]*w), int(anchor[1]*h), int(anchor[2]*w), int(anchor[3]*h)] for anchor in pdefined_anchors]
    input_img_score = scores_list[0]

    # update scores_list, bboxes_list
    scores_list = scores_list[1:]
    bboxes_list = bboxes_list[1:]

    # sort
    idx_sorted = np.argsort(-np.array(scores_list))
    scores_list_sorted = [scores_list[i] for i in idx_sorted]
    bboxes_list_sorted = [bboxes_list[i] for i in idx_sorted]
    scores_nms, bboxes_nms, idx = bboxes_overlap_check(scores_list_sorted, bboxes_list_sorted, nms_threshold=0.65)
    print("input_img_score:", input_img_score)
    print("scores_nms:", scores_nms)
    return image, scores_nms[0:3], bboxes_nms[0:3]


app = Flask(__name__)
@app.route('/sample', methods=['POST'])
def main():
    global device, VPN_net, pdefined_anchors, image_list, transform
    for i in range(9):
        img = Image.open(BytesIO(request.get_data().split(b'\r\n--')[i+1].split(b'\r\n\r\n')[1])).convert('RGB')
        image_list.append(img)

    # 3つのリストを定義（オリジナル画像、スコア、バウンディングボックス）
    img_li, score_li, bbox_li = [], [], []
    start = time.time()
    for image in image_list:
        img, score, bbox = predict(image)
        for _ in range(3):
            img_li.append(img)
        score_li += score
        bbox_li += bbox
    end = time.time()
    print("diff:", end-start)
    print("score:", score)
    print("bbox:", bbox)
    print("ok!!")

    # TOP3の画像を返す
    response = []
    # print("score_li:", score_li)
    sorted_score = sorted(score_li)
    # print("sorted_score:", sorted_score)
    for i in range(3):
        idx = score_li.index(sorted_score[-(i+1)])
        comp = img_li[idx].crop(bbox_li[idx])
        img_bytes = BytesIO() # BytesIO -> メモリ上でバイナリデータを扱うための機能
        comp.save(img_bytes, format='jpeg') #ファイル名ではなく、バイナリデータを渡す
        img_str = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
        response.append({"id": i, "result": img_str})
    # return jsonify(response)

    # Unityで処理しやすい形に変換 ==> {"Key(以下ではrecommends)":[{key:value, key:value},...,{key:value, key:value}]}
    response_unity = {"recommends": response}
    return jsonify(response_unity)


if __name__ == '__main__':
    load_model("./model_pth/VRPhoto/VPN_yoko_20221219.pth")
    pdefined_anchors = pd.read_pickle("./config/anchor_square_500.pkl")
    transform = val_transform(320)
    print("---------- starting server ----------")
    app.run(host='0.0.0.0', port=5004)
    # main()


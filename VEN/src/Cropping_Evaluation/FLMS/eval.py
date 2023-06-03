import os
import glob
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import pandas as pd
import numpy as np
import progressbar
import json
import sys
sys.path.append("../../nets")
from FullVggCompositionNet import FullVggCompositionNet as CompositionNet
from SiameseNet import SiameseNet


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

    keep_bboxes = np.ones(len(scores), dtype=np.bool)
    for i in range(len(scores)-1):
        if keep_bboxes[i]:
            overlap = bboxes_jaccard(bboxes[i], bboxes[(i+1):])
            keep_overlap = (overlap < nms_threshold)
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)
    idxes = np.where(keep_bboxes)
    return scores[idxes].tolist(), bboxes[idxes].tolist(), idxes


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    ar1 = (a[2]-a[0])*(a[3]-a[1])
    ar2 = (b[2]-b[0])*(b[3]-b[1])
    ov = (x2-x1)*(y2-y1)
    if x2 > x1 and y2 > y1:
        iou = float(ov) / (ar1+ar2-ov)
        return iou
    else:
        return 0


def disp_error(bbox_a, bbox_b, img_w, img_h):
    # normalize (if index is even, divide by img_w, elif index is odd, divide by img_h)
    a = list(map(lambda x, idx: x/img_w if idx % 2 == 0 else x/img_h, bbox_a, range(4)))
    b = list(map(lambda x, idx: x/img_w if idx % 2 == 0 else x/img_h, bbox_b, range(4)))
    left = np.abs(a[0]-b[0])
    right = np.abs(a[2]-b[2])
    top = np.abs(a[1]-b[1])
    down = np.abs(a[3]-b[3])
    disp_error = (left + right + top + down) / 4
    return disp_error


def offset_error(a, b):
    center_x1 = (a[2]-a[0])/2
    center_y1 = (a[3]-a[1])/2
    center_x2 = (b[2]-b[0])/2
    center_y2 = (b[3]-b[1])/2
    offset_error = np.sqrt((center_x1-center_x2) ** 2 + (center_y1-center_y2) ** 2)
    return offset_error


def val_transform(image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform


def main():
    parser = argparse.ArgumentParser(description="VEN")
    #parser.add_argument('--model_path', type=str, default='../../config/EvaluationNet.pth.tar')
    parser.add_argument('--model_path', type=str, default='../../model_pth/VRPhoto/VEN_20221206.pth')
    args = parser.parse_args()

    # load anchor
    pdefined_anchors = pd.read_pickle('../../config/anchor_square_500.pkl')
    pdefined_anchors = pdefined_anchors[1:]
    print("len(pdefined_anchors):", len(pdefined_anchors))
    image_list = glob.glob('./image/*')
    #image_list = image_list[0:1]
    transform = val_transform(224)
    with open("500_image_dataset.json") as f:
        str1 = f.read()
    annotation_dict = json.loads(str1)

    # model define
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    single_pass_net = CompositionNet(pretrained=False, LinearSize1=1024, LinearSize2=512).to(device)
    siamese_net = SiameseNet(single_pass_net)
    #siamese_net.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage)['state_dict'])
    siamese_net.load_state_dict(torch.load(args.model_path))
    siamese_net.eval()

    # eval list
    iou_scores_list = []
    disp_errors_list = []
    offset_errors_list = []

    # processing per image
    for img_idx, img_path in enumerate(image_list):
        img_crops, img_bboxes, w, h = GetImageCrops(img_path, pdefined_anchors)
        img_name = os.path.basename(img_path)
        print(f"[{img_idx+1} | {len(image_list)}]\t{img_name}")
        pbar = progressbar.ProgressBar(max_value=len(img_crops))
        scores_list = []
        bboxes_list = []
        # processing per crop image
        for crop_idx, (img_crop, img_bbox) in enumerate(zip(img_crops, img_bboxes)):
            pbar.update(crop_idx)
            #inp = Variable(transform(img_crop).to(device))
            inp = transform(img_crop).to(device)
            oup = single_pass_net(inp.unsqueeze(0))  # inp.unsqueeze(0): (3,224,224) -> (1,3,224,224)
            scores_list.append(float(oup.data.cpu().numpy()[0][0]))
            bboxes_list.append(img_bbox)

        idx_sorted = np.argsort(-np.array(scores_list))
        scores_list_sorted = [scores_list[i] for i in idx_sorted]
        bboxes_list_sorted = [bboxes_list[i] for i in idx_sorted]
        scores_nms, bboxes_nms, idx = bboxes_overlap_check(scores_list_sorted, bboxes_list_sorted, nms_threshold=0.6)

        iou_list = []
        disp_list = []
        offset_list = []
        for bbox_pred in bboxes_nms[0:5]: #TOP5
            for bbox_gt_raw in annotation_dict[img_name]:
                if min(bbox_gt_raw) >= 0:
                    # (height_min, width_min, height_max, width_max) -> (width_min, height_min, width_max, height_max)
                    bbox_gt = []
                    bbox_gt.append(bbox_gt_raw[1])
                    bbox_gt.append(bbox_gt_raw[0])
                    bbox_gt.append(bbox_gt_raw[3])
                    bbox_gt.append(bbox_gt_raw[2])
                    #print(f"bbox_pred: {bbox_pred}, bbox_gt: {bbox_gt}")
                    iou_list.append(iou(bbox_pred, bbox_gt))
                    disp_list.append(disp_error(bbox_pred, bbox_gt, w, h))
                    offset_list.append(offset_error(bbox_pred, bbox_gt))
        iou_scores_list.append(max(iou_list))
        disp_errors_list.append(min(disp_list))
        offset_errors_list.append(min(offset_list))


    print("TOP5")
    print("iou_scores_list:", iou_scores_list)
    print("disp_errors_list:", disp_errors_list)
    print("offset_errors_list:", offset_errors_list)
    iou_mean = np.mean(iou_scores_list)
    disp_mean = np.mean(disp_errors_list)
    offset_mean = np.mean(offset_errors_list)
    print("iou_mean:", iou_mean)
    print("disp_mean:", disp_mean)
    print("offset_mean:", offset_mean)

    csvfile_savedir = get_dir('./output')
    df = pd.DataFrame({'IOU_mean': [iou_mean], 'DispError_mean': [disp_mean], 'OffsetError_mean': [offset_mean]}).T
    #df.to_csv(f"{csvfile_savedir}/VEN_VRPhoto_20221206_top5.csv", header=None)


if __name__ == '__main__':
    main()

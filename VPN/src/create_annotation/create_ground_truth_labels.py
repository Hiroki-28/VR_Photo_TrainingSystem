import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import os, sys, glob
import json
import progressbar
import numpy as np
import pandas as pd
from PIL import Image
from FullVggCompositionNet import FullVggCompositionNet as CompositionNet
from SiameseNet import SiameseNet
from multiprocessing import Pool
import multiprocessing as multi

global pdefined_anchors


# custom dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform, img_path):
        self.img_dir = img_dir
        #self.filelist = glob.glob(os.path.join(self.img_dir, "*"))
        self.transform = transform
        self.filelist = [img_path]
        n_cores = int(multi.cpu_count()/2)
        p = Pool(n_cores)
        print('start')
        res = p.map(GetImgCrops, self.filelist)
        #print("res:", res)
        print('ok')

        self.crops = []
        self.bboxes = []
        self.names = []
        self.crops += res[0][0]
        self.bboxes += res[0][1]
        self.names += res[0][2]
    
    def __getitem__(self, index):
        return self.transform(self.crops[index]), torch.Tensor(self.bboxes[index]), self.names[index]

    def __len__(self):
        return len(self.crops)


def valid_transform(image_size=224):
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return val_transform


def GetImgCrops(image_path):
    global pdefined_anchors
    img_name = os.path.basename(image_path)

    if isinstance(image_path, str):
        image = Image.open(image_path)
    w, h = image.size
    img_crops = []
    img_bboxes = []
    img_names = []
    for anchor in pdefined_anchors:
        x1, y1, x2, y2 = int(anchor[0]*w), int(anchor[1]*h), int(anchor[2]*w), int(anchor[3]*h)
        img_crop = image.crop((x1,y1,x2,y2)).copy()
        img_crops.append(img_crop)
        img_bboxes.append([x1,y1,x2,y2])
        img_names.append(img_name)
    return img_crops, img_bboxes, img_names


def main():
    global pdefined_anchors
    # parser
    parser = argparse.ArgumentParser(description="VEN")
    parser.add_argument('--l1', default=1024, type=int)
    parser.add_argument('--l2', default=512, type=int)
    parser.add_argument('--resume', default='../config/VEN_20221206.pth', type=str)
    parser.add_argument('--save_file', default='./output_json/VRPhoto_tate_500.json', type=str)
    parser.add_argument('--dataset_dir', default='../dataset/VRPhoto/images_tate_250', type=str)
    #parser.add_argument('--resume', default='../config/EvaluationNet.pth.tar', type=str)
    #parser.add_argument('--save_file', default='./output_json/CPC_label_500.json', type=str)
    #parser.add_argument('--dataset_dir', default='../dataset/CPC', type=str)
    args = parser.parse_args()

    pdefined_anchors = pd.read_pickle('../config/anchor_square_500.pkl')
    image_list = glob.glob(f'{args.dataset_dir}/*.jpg')
    image_annotation = [{} for _ in range(len(image_list))]

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    ckpt_file = args.resume
    single_pass_net = CompositionNet(pretrained=False, LinearSize1=args.l1, LinearSize2=args.l2)
    siamese_net = SiameseNet(single_pass_net).to(device)
    siamese_net.load_state_dict(torch.load(ckpt_file))
    #ckpt = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
    #siamese_net.load_state_dict(ckpt['state_dict'])
    single_pass_net.eval()

    # create the ground truth labels
    for idx, img_path in enumerate(image_list):
        print(f"[{idx+1}/{len(image_list)}]")
        dataset = Dataset(img_dir=args.dataset_dir, transform=valid_transform(224), img_path=img_path)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False, num_workers=4)
        for img, bbox, name in data_loader:
            img = img.to(device)
            oup = single_pass_net(img)
            print("bbox:", bbox)
            image_annotation[idx]['img_name'] = name[0]
            image_annotation[idx]['score'] = [float(value) for li in oup.data.cpu().numpy() for value in li]
            image_annotation[idx]['bboxes'] = bbox.numpy().astype(int).tolist()
            del img, oup, bbox, name
            torch.cuda.empty_cache()

    # save json
    save_file = args.save_file
    print("Done Computing, saving to {:s}".format(save_file))
    with open(save_file, 'w') as f:
        json.dump(image_annotation, f, indent=2)
    print("ok!")


if __name__ == "__main__":
    main()


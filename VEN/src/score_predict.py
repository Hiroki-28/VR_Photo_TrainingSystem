import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import os, sys, glob, json
import numpy as np
import pandas as pd
from PIL import Image
from nets.FullVggCompositionNet import FullVggCompositionNet as CompositionNet
from nets.SiameseNet import SiameseNet

def get_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    return directory


def val_transform(image_size=224):
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return valid_transform


def main():
    parser = argparse.ArgumentParser(description='VEN')
    parser.add_argument("--resume", type=str, default='./model_pth/VRPhoto/VEN_20221206.pth')
    parser.add_argument("--dataset_dir", type=str, default='./datasets/VRPhoto/OriginalImages/')
    args = parser.parse_args()
    ckpt_file = args.resume

    # dataset
    image_list = glob.glob(f'{args.dataset_dir}/*/*.jpg')
    image_annotation = [{} for _ in range(len(image_list))]
    scores_list = []

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    transform = val_transform(224)
    single_pass_net = CompositionNet(pretrained=False, LinearSize1=1024, LinearSize2=512)
    siamese_net = SiameseNet(single_pass_net).to(device)
    siamese_net.load_state_dict(torch.load(ckpt_file))
    single_pass_net.eval()

    for idx, img_path in enumerate(image_list):
        print(f'[{idx+1} | {len(image_list)}]')
        img = Image.open(img_path)
        img = transform(img).to(device)
        inp = img.unsqueeze(0)
        oup = single_pass_net(inp)
        image_annotation[idx]['img_path'] = img_path
        image_annotation[idx]['score'] = float(oup.data.cpu().numpy()[0][0])
        scores_list.append(oup.data.cpu().numpy()[0][0])

    save_dir = get_dir('./ScoreResult')
    save_jsonfile = f'{save_dir}/composition_score.json'
    save_csvfile = f'{save_dir}/composition_score.csv'
    
    # output jsonfile
    with open(save_jsonfile, 'w') as f:
        json.dump(image_annotation, f, indent=2)

    df = pd.Series(scores_list)
    df.to_csv(save_csvfile, header=None, index=None)


if __name__ == "__main__":
    main()

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import os, sys, glob, json
import numpy as np
import pandas as pd
from PIL import Image
from net import ResNet_GatedCNN
from net import ColorHarmonyNet

def get_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    return directory


def val_transform(image_size=224):
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms.Normalize(mean=[0.3876, 0.3853, 0.3558], std=[0.2125, 0.1957, 0.1948])])
    return valid_transform


def calc_probability(net, patches, patch_num_edge, device):
    #labels = labels.to(device).to(torch.long)
    #labels = torch.eye(8)[labels]  # 8 -> class_num
    for i in range(1, patch_num_edge-1):
        for j in range(1, patch_num_edge-1):
            patches_center = patches[:, i, j, :, :, :]       # center, [32, 3, 16, 16]
            patches_neighbor1 = patches[:, i-1, j, :, :, :]  # upper , [32, 3, 16, 16]
            patches_neighbor2 = patches[:, i, j-1, :, :, :]  # left  , [32, 3, 16, 16]
            patches_neighbor3 = patches[:, i, j+1, :, :, :]  # right , [32, 3, 16, 16]
            patches_neighbor4 = patches[:, i+1, j, :, :, :]  # lower , [32, 3, 16, 16]
            # calcurate sum of the probability
            prob = net(patches_center, patches_neighbor1, patches_neighbor2, patches_neighbor3, patches_neighbor4)  # without  label
            #prob = net(patches_center, patches_neighbor1, patches_neighbor2, patches_neighbor3, patches_neighbor4, labels) # with label
            if i == 1 and j == 1:
                prob_cat = prob
            else:
                prob_cat = torch.cat((prob_cat, prob),1)

    # calcurate the probability (batch size)
    prob_sum = torch.sum(prob_cat, 1)
    #patch_num = torch.ones(prob_sum.numel()).to(device) * ((patch_num_edge-2)**2)  # if patch_size==16 -> [144,144, ,,, ,144]
    #probs = prob_sum / patch_num
    #return probs
    return prob_sum


def main():
    parser = argparse.ArgumentParser(description='VEN')
    parser.add_argument("--resume", type=str, default='../model_pth/VRPhoto/CHE_20221204.pth')
    parser.add_argument("--dataset_dir", type=str, default='../dataset/VRPhoto/all')
    args = parser.parse_args()
    ckpt_file = args.resume
    # create instance called "transform"
    transform = val_transform(224)
    patch_size = 16
    patch_num_edge = int(224 / patch_size)

    # dataset
    image_list = glob.glob(f'{args.dataset_dir}/*/*.jpg')
    image_annotation = [{} for _ in range(len(image_list))]
    scores_list = []

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    color_net = ResNet_GatedCNN().to(device)
    color_harmony_net = ColorHarmonyNet(color_net).to(device)
    color_harmony_net.load_state_dict(torch.load(args.resume))
    color_harmony_net.eval()
    

    for idx, img_path in enumerate(image_list):
        print(f'[{idx+1} | {len(image_list)}]')
        img = Image.open(img_path)
        # transform(img) -> call fauction
        img = transform(img).to(device).unsqueeze(0)  # shape: (3, 224, 224) -> (1, 3, 224, 224)
        patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute([0, 2, 3, 1, 4, 5])
        oup = calc_probability(color_harmony_net, patches, patch_num_edge, device)
        image_annotation[idx]['img_name'] = os.path.basename(img_path)
        image_annotation[idx]['score'] = float(oup.data.cpu().numpy()[0])
        scores_list.append(oup.data.cpu().numpy()[0])

    save_dir = get_dir('./ScoreResult')
    save_jsonfile = f'{save_dir}/color_score.json'
    save_csvfile = f'{save_dir}/color_score.csv'
    
    # output jsonfile
    with open(save_jsonfile, 'w') as f:
        json.dump(image_annotation, f, indent=2)

    df = pd.Series(scores_list)
    df.to_csv(save_csvfile, header=None, index=None)


if __name__ == "__main__":
    main()

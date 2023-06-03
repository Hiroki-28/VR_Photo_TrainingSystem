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
from net import ResNet, ResNet_GatedCNN, ColorHarmonyNet
from net_with_label import ColorHarmonyNet_w_labels


def valid_transform(image_size=224):
    valid_aug_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transforms.Normalize(mean=[0.3876, 0.3853, 0.3558], std=[0.2125, 0.1957, 0.1948])])
    return valid_aug_transform


def get_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path


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
    patch_num = torch.ones(prob_sum.numel()).to(device) * ((patch_num_edge-2)**2)  # if patch_size==16 -> [144,144, ,,, ,144]
    probs = prob_sum / patch_num
    return probs


def main():
    # parser setting
    parser = argparse.ArgumentParser(description="Color Harmony model trained on AVA")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--data_folder_test', type=str, default='../dataset/VRPhoto/data_811/test')
    args = parser.parse_args()

    # load dataset and DataLoader
    test_dataset = datasets.ImageFolder(root=args.data_folder_test, transform=valid_transform(224))
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # define model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    color_net = ResNet_GatedCNN().to(device)
    color_harmony_net = ColorHarmonyNet(color_net).to(device)

    # loss function
    criterion = nn.BCELoss()

    ### patch
    patch_size = args.patch_size
    patch_num_edge = int(224 / patch_size)
    # create test metrics list
    test_precisions = []
    test_recalls    = []
    test_f_measures = []
    test_accuracys  = []

    # load model
    color_harmony_net.load_state_dict(torch.load('../model_pth/VRPhoto/CHE_20221204.pth'))
    print("len(test_loader):", len(test_loader))
        
    #===== switch to eval mode =====
    color_harmony_net.eval()
    test_loss = 0
    positive_count = 0
    true_count = 0
    total_count = 0
    true_positive_count = 0
    true_negative_count = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.to(device).to(torch.float32)
            labels = labels.to(device).to(torch.float32)
            ### calculate the probability of being good aesthetic
            patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            patches = patches.permute([0, 2, 3, 1, 4, 5])
            probs   = calc_probability(color_harmony_net, patches, patch_num_edge, device)
                
            # calcurate loss value
            loss = criterion(probs, labels)
            test_loss += loss.item()

            # mesuring metrics
            labels = torch.where(labels == 1, True, False)
            preds = torch.where(probs >= 0.5, True, False)
            tp = labels & preds
            tn = ~labels & ~preds
            true_positive_count += torch.where(tp==True)[0].numel()
            true_negative_count += torch.where(tn==True)[0].numel()
            true_count += torch.where(labels==True)[0].numel()
            positive_count += torch.where(preds==True)[0].numel()
            total_count += labels.numel()
    
    print("true_positive_count:", true_positive_count)
    print("true_negative_count:", true_negative_count)
    print("true_count:", true_count)
    print("positive_count:", positive_count)
    print("total_count:", total_count)

    # calculate average loss per epoch
    avg_test_loss = test_loss / len(test_loader)
        
    # calculate precision, recall, F-measure, accuracy
    precision = float(true_positive_count / positive_count)
    recall = float(true_positive_count / true_count)
    f_measure = float(2 * precision * recall / (precision + recall))
    accuracy = float((true_positive_count + true_negative_count) / total_count)
    print("test_F_measure: {0:.4f}, test_acc: {1:.4f}, test_loss: {2:.4f}".format(f_measure, accuracy, avg_test_loss))

    # add to list
    test_precisions.append(precision)
    test_recalls.append(recall)
    test_f_measures.append(f_measure)
    test_accuracys.append(accuracy)


    # output loss transition and metrics transition
    df = pd.DataFrame({'precision': test_precisions, 'recall': test_recalls, 'f_measure': test_f_measures, 'accuracy': test_accuracys})
    #df.to_csv("../output/VRPhoto/CHE_test_20221204.csv", index=None)


if __name__ == '__main__':
    main()

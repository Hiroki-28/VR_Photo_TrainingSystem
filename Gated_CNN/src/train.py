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
#from net_with_label import ColorHarmonyNet_w_labels
import sys


class EarlyStopping:
    def __init__(self, patience, path, best_loss):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_loss = best_loss
        self.early_stop = False

    def __call__(self, model, valid_loss):
        if self.best_loss < valid_loss:
            self.counter += 1
            print("EarlyStopping counter: {0} out of {1}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.checkpoint(model, valid_loss)
            self.counter = 0
    
    def checkpoint(self, model, valid_loss):
        print("Validation loss decreased ({0:.4f} ==> {1:.4f})".format(self.best_loss, valid_loss))
        torch.save(model.state_dict(), self.path)
        self.best_loss = valid_loss


def train_transform(image_size=224):
    tr_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transforms.Normalize(mean=[0.3876, 0.3853, 0.3558], std=[0.2125, 0.1957, 0.1948])])
    return tr_transform


def valid_transform(image_size=224):
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transforms.Normalize(mean=[0.3876, 0.3853, 0.3558], std=[0.2125, 0.1957, 0.1948])])
    return val_transform


def get_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path


def load_loss_li(filepath):
    df = pd.read_csv(filepath)
    train_loss_li = list(df['loss'])
    valid_loss_li = list(df['val_loss'])
    epoch_li = list(df['epoch'])
    return train_loss_li, valid_loss_li, epoch_li


def load_eval_li(filepath):
    df = pd.read_csv(filepath)
    precision_li = list(df['precision'])
    recall_li = list(df['recall'])
    f_measure_li = list(df['f_measure'])
    accuracy_li = list(df['accuracy'])
    return precision_li, recall_li, f_measure_li, accuracy_li


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
            if i == 1 and j == 1:
                prob_cat = prob
            else:
                prob_cat = torch.cat((prob_cat, prob),1)

            """
            if i == 1 and j == 1:
                prob1 = prob
            elif i == 1 and j == 2:
                prob2 = prob
            elif i == 1 and j == 3:
                print("prob1:", prob1)
                print("prob2:", prob2)
                prob3 = prob1 + prob2
                print("prob3:", prob3)
                break
            """

    # calcurate the probability (batch size)
    prob_sum = torch.sum(prob_cat, 1)
    patch_num = torch.ones(prob_sum.numel()).to(device) * ((patch_num_edge-2)**2)  # if patch_size==16 -> [144,144, ,,, ,144]
    probs = prob_sum / patch_num
    return probs


def main():
    # parser setting
    parser = argparse.ArgumentParser(description="Color Harmony model")
    parser.add_argument('--dataset_tr_dir', default='../dataset/VRPhoto/data_811/train', type=str)
    parser.add_argument('--dataset_val_dir', default='../dataset/VRPhoto/data_811/valid', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--step_size', default=300, type=int, help='the cycle to change learning rate')
    parser.add_argument('--gamma', default=0.1, type=float, help='the ratio when the learning rate changes')
    parser.add_argument('--model_savedir', default='../model_pth/VRPhoto', type=str)
    parser.add_argument('--csvfile_savedir', default='../output/VRPhoto', type=str)
    parser.add_argument('--date', default='20221216', type=str)
    parser.add_argument('--patience', default=300, type=int)
    parser.add_argument('--resume', default=None, type=str, help = 'trained filepath(model_pth)')
    parser.add_argument('--save_point', default=False, type=bool)
    args = parser.parse_args()

    # create folder and save hyper parameter
    model_savedir = get_dir(args.model_savedir)
    csvfile_savedir = get_dir(args.csvfile_savedir)
    df = pd.DataFrame({'dataset_tr_dir': [args.dataset_tr_dir], 'dataset_val_dir': [args.dataset_val_dir], 'batch_size': [args.batch_size], 'learning_rate': [args.learning_rate], 'momentum': [args.momentum], 'weight_decay': [args.weight_decay], 'step_size': [args.step_size], 'gamma': [args.gamma]}).T
    df.to_csv(f"{csvfile_savedir}/CHE_config_{args.date}.csv", header=None)

    # create dataset
    train_dataset = datasets.ImageFolder(root=args.dataset_tr_dir, transform=train_transform(224))
    valid_dataset = datasets.ImageFolder(root=args.dataset_val_dir, transform=valid_transform(224))

    # load DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # define model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    color_net = ResNet_GatedCNN().to(device)
    color_harmony_net = ColorHarmonyNet(color_net).to(device)
    #color_harmony_net = ColorHarmonyNet_w_labels(color_net).to(device)

    # setting loss function and optimization function
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(color_harmony_net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(color_harmony_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    earlystopping = EarlyStopping(patience=args.patience, path=os.path.join(model_savedir, f'CHE_{args.date}.pth'), best_loss=np.inf)

    ### train
    num_epochs = args.num_epochs
    patch_size = args.patch_size
    patch_num_edge = int(224 / patch_size)

    # create list
    #init_list = ['train_loss_list', 'valid_loss_list', 'epoch_list', 'train_precisions', 'train_recalls', 'train_f_measures', 'train_accuracies', 'valid_precisions', 'valid_recalls', 'valid_f_measures', 'valid_accuracies']
    #for name in init_list:
    #    exec("{0}=[]".format(name))
    train_loss_list, valid_loss_list, epoch_list = [], [], []
    train_precisions, train_recalls, train_f_measures, train_accuracies = [], [], [], []
    valid_precisions, valid_recalls, valid_f_measures, valid_accuracies = [], [], [], []

    if args.save_point:
        train_loss_list, valid_loss_list, epoch_list = load_loss_li(f"{csvfile_savedir}/CHE_loss_{args.date}.csv")
        train_precisions, train_recalls, train_f_measures, train_accuracies = load_eval_li(f"{csvfile_savedir}/CHE_train_{args.date}.csv")
        valid_precisions, valid_recalls, valid_f_measures, valid_accuracies = load_eval_li(f"{csvfile_savedir}/CHE_valid_{args.date}.csv")
        color_harmony_net.load_state_dict(torch.load(args.resume))
        earlystopping = EarlyStopping(patience=args.patience, path=os.path.join(model_savedir, 'CHE_{args.date}.pth'), best_loss=min(valid_loss_list))
    
    print("len(train_loader):", len(train_loader))
    print("len(valid_loader):", len(valid_loader))
    for epoch in range(0, num_epochs):
        print("{}epoch".format(epoch+1))
        train_loss = 0
        positive_count = 0
        true_count = 0
        total_count = 0
        true_positive_count = 0
        true_negative_count = 0

        #===== switch to train mode =====
        color_harmony_net.train()
        # load batch data from train_loader
        for idx, (images, labels) in enumerate(train_loader):
            if (idx + 1) % 10 == 0:
                print("{0}th batch".format(idx+1))
            
            images = images.to(device).to(torch.float32)
            labels = labels.to(device).to(torch.float32)  # bad -> 0, good -> 1 (torchvision.datasets.folder)
            # initialize gradient
            optimizer.zero_grad()
            ### calculate the probability of being good aesthetic
            patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)  # unfold-> convert image to patches, ex-> [32, 3, 14, 14, 16, 16]
            patches = patches.permute([0, 2, 3, 1, 4, 5])  # ex-> [32, 14, 14, 3, 16, 16] (batch, patch_num(tate,yoko), rgb, patch_size(tate, yoko))
            
            probs   = calc_probability(color_harmony_net, patches, patch_num_edge, device)
            
            # update weight
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
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
            
            
        
        # calculate average loss per epoch
        avg_train_loss = train_loss / len(train_loader)

        # calculate precision, recall, F-measure, accuracy
        precision = float(true_positive_count / positive_count)
        recall = float(true_positive_count / true_count)
        f_measure = float(2 * precision * recall / (precision + recall))
        accuracy = float((true_positive_count + true_negative_count) / total_count)
        print("Epoch [{0}/{1}], F_measure: {2:.4f}, acc: {3:.4f}, loss: {4:.4f}".format(epoch+1, num_epochs, f_measure, accuracy, avg_train_loss))
        
        # add to list
        train_loss_list.append(avg_train_loss)
        train_precisions.append(precision)
        train_recalls.append(recall)
        train_f_measures.append(f_measure)
        train_accuracies.append(accuracy)


        #===== switch to eval mode =====
        color_harmony_net.eval()
        valid_loss = 0
        positive_count = 0
        true_count = 0
        total_count = 0
        true_positive_count = 0
        true_negative_count = 0
        with torch.no_grad():
            for idx, (images, labels) in enumerate(valid_loader):
                if (idx + 1) % 10 == 0:
                    print("{0}th batch".format(idx+1))
                
                images = images.to(device).to(torch.float32)
                labels = labels.to(device).to(torch.float32)
                ### calculate the probability of being good aesthetic
                patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
                patches = patches.permute([0, 2, 3, 1, 4, 5])
                probs   = calc_probability(color_harmony_net, patches, patch_num_edge, device)
                
                # calcurate loss value
                loss = criterion(probs, labels)
                valid_loss += loss.item()

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
        

        # calculate average loss per epoch
        avg_valid_loss = valid_loss / len(valid_loader)
        
        # calculate precision, recall, F-measure, accuracy
        precision = float(true_positive_count / positive_count)
        recall = float(true_positive_count / true_count)
        f_measure = float(2 * precision * recall / (precision + recall))
        accuracy = float((true_positive_count + true_negative_count) / total_count)
        print("Epoch [{0}/{1}], val_F_measure: {2:.4f}, val_acc: {3:.4f}, val_loss: {4:.4f}".format(epoch+1, num_epochs, f_measure, accuracy, avg_valid_loss))

        # add to list
        valid_loss_list.append(avg_valid_loss)
        valid_precisions.append(precision)
        valid_recalls.append(recall)
        valid_f_measures.append(f_measure)
        valid_accuracies.append(accuracy)
        epoch_list.append(epoch+1)

        # earlystopping
        earlystopping(color_harmony_net, avg_valid_loss)
        if earlystopping.early_stop:
            print("Early Stopping!")
            break

        # reduce learning_rate
        scheduler.step()

        # output loss transition and metrics transition
        df = pd.DataFrame({'epoch': epoch_list, 'loss': train_loss_list, 'val_loss': valid_loss_list})
        df2 = pd.DataFrame({'epoch': epoch_list, 'precision': train_precisions, 'recall': train_recalls, 'f_measure': train_f_measures, 'accuracy': train_accuracies})
        df3 = pd.DataFrame({'epoch': epoch_list, 'precision': valid_precisions, 'recall': valid_recalls, 'f_measure': valid_f_measures, 'accuracy': valid_accuracies})
        df.to_csv(f"{csvfile_savedir}/CHE_loss_{args.date}.csv", index=None)
        df2.to_csv(f"{csvfile_savedir}/CHE_train_{args.date}.csv", index=None)
        df3.to_csv(f"{csvfile_savedir}/CHE_valid_{args.date}.csv", index=None)


if __name__ == '__main__':
    main()

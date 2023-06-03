import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import os, sys, glob
import json
import numpy as np
import pandas as pd
from PIL import Image
from net import SSD_320 as ssd_320


# custom dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform, annotation_path):
        self.img_dir = img_dir
        self.transform = transform
        with open(annotation_path) as f:
            json_file = f.read()
        self.label_dict = json.loads(json_file)

    def __getitem__(self, index):
        img_data = Image.open(os.path.join(self.img_dir, self.label_dict[index]['img_name']))
        img_data = self.transform(img_data)
        score_list = torch.Tensor(self.label_dict[index]['score'])
        return img_data, score_list

    def __len__(self):
        return len(self.label_dict)


# custom loss function
class MPSELoss(nn.Module):
    def __init__(self):
        super(MPSELoss, self).__init__()

    def forward(self, outputs, targets):
        count = 0
        for idx, (output, target) in enumerate(zip(outputs, targets)):
            for i in range(len(output)):
                for j in range(i+1, len(output)):
                    if idx == 0 and i == 0 and j == 1:
                        loss = ((output[i]-output[j]) - (target[i]-target[j])) ** 2
                    else:
                        loss += ((output[i]-output[j]) - (target[i]-target[j])) ** 2
                        print("loss:", loss)
                    #del output[i]
                    count += 1
            del output, target
        loss = loss / count

        # NG method -> When I create a new list, the gradient does not update correctly
        #pairwise_square_error = []
        #for output, target in zip(outputs, targets):
        #    for i in range(len(output)):
        #        for j in range(i+1, len(output)):
        #            diff = ((output[i]-output[j]) - (target[i]-target[j])) ** 2
        #            pairwise_square_error.append(diff)

        #loss = torch.mean(torch.Tensor(pairwise_square_error)) 
        #loss = Variable(loss, requires_grad=True)
        return loss


# RMSELoss
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))


# early stopping
class EarlyStopping:
    def __init__(self, patience, path):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, model, valid_loss):
        if self.best_loss < valid_loss:
            self.counter += 1
            print("EarlyStopping counter: {0}/{1}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.checkpoint(model, valid_loss)
            self.counter = 0
    def checkpoint(self, model, valid_loss):
        print("Validation loss decreased ({0:.4f} ==> {1:.4f})".format(self.best_loss, valid_loss))
        torch.save(model.state_dict(), self.path)
        self.best_loss = valid_loss


def get_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    return directory


def train_transform(image_size=320):
    train_aug_transform = transforms.Compose([
        transforms.Resize(size=(image_size, image_size), interpolation=Image.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return train_aug_transform


def valid_transform(image_size=320):
    valid_aug_transform = transforms.Compose([
        transforms.Resize(size=(image_size, image_size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return valid_aug_transform


def main():
    # parser
    parser = argparse.ArgumentParser(description="VPN")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_epochs', default=400, type=int)
    parser.add_argument('--patience', default=15, type=int)
    parser.add_argument('--date', default='20221225', type=str)
    parser.add_argument('--dataset_dir', default='./dataset/VRPhoto/images_tate_250', type=str)
    parser.add_argument('--annotation_path', default='./create_annotation/output_json/VRPhoto_tate_500.json', type=str, help='label information')
    parser.add_argument('--param_savedir', default='./model_pth/VRPhoto', type=str)
    parser.add_argument('--csvfile_savedir', default='./output/VRPhoto', type=str)
    parser.add_argument('--resume', default='./model_pth/CPC/VPN_20221210.pth', type=str)
    args = parser.parse_args()

    # dataset
    dataset = Dataset(img_dir=args.dataset_dir, transform=train_transform(320), annotation_path=args.annotation_path)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)], generator=torch.Generator().manual_seed(0))
    # data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    VPN_net = ssd_320(num_classes = 500).to(device)
    ckpt_file = args.resume
    VPN_net.load_state_dict(torch.load(ckpt_file))
    
    #criterion = MPSELoss()  # loss function
    criterion = RMSELoss()
    #criterion = nn.MSELoss()
    #criterion = nn.PairwiseDistance(p=2, eps=0.0)
    optimizer = torch.optim.SGD(VPN_net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)  # optimization function
    #optimizer = torch.optim.AdamW(VPN_net.parameters(), lr=1e-3, weight_decay=1e-6)  # optimization function
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.6)  # learning schedule instance
    param_savedir = get_dir(args.param_savedir)
    csvfile_savedir = get_dir(args.csvfile_savedir)
    date = args.date
    earlystopping = EarlyStopping(patience=args.patience, path=os.path.join(param_savedir, 'VPN_{}.pth'.format(date)))

    # model learning
    num_epochs = args.num_epochs
    train_loss_list = []
    valid_loss_list = []
    epoch_list = []
    print("len(train_loader):", len(train_loader))
    print("len(valid_loader):", len(valid_loader))
    for epoch in range(num_epochs):
        print("{}epoch".format(epoch+1))
        # initialize every epoch
        train_loss = 0
        valid_loss = 0
        
        # ===== train =====
        VPN_net.train()
        batch_count = 0
        for img, score_list in train_loader:
            img, score_list = img.to(device), score_list.to(device)
            # initialize grad
            optimizer.zero_grad()
            # forward propagation
            outputs = VPN_net(img)
            #loss = criterion(outputs, score_list)
            loss = torch.mean(criterion(outputs, score_list))  # PairwiseDistance
            train_loss += loss.item()
            # back propagation
            loss.backward()
            optimizer.step()
            # batch_count
            batch_count += 1
            if batch_count % 20 == 0:
                print("batch_count:{0}, loss:{1}".format(batch_count, loss))
        # calculate avg_train_loss every 1epoch
        avg_train_loss = train_loss / len(train_loader)

        # ===== valid =====
        VPN_net.eval()
        with torch.no_grad():
            for img, score_list in valid_loader:
                img, score_list = img.to(device), score_list.to(device)
                # forward propagation
                outputs = VPN_net(img)
                #loss = criterion(outputs, score_list)
                loss = torch.mean(criterion(outputs, score_list))  # PairwiseDistance
                valid_loss += loss.item()
        # calculate avg_valid_loss every 1epoch
        avg_valid_loss = valid_loss / len(valid_loader)
        print("Epoch [{0}/{1}], loss: {2:.4f}, val_loss: {3:.4f}".format(epoch+1, num_epochs, avg_train_loss, avg_valid_loss))
        
        # add to loss_list
        epoch_list.append(epoch+1)
        train_loss_list.append(avg_train_loss)
        valid_loss_list.append(avg_valid_loss)

        # call function
        earlystopping(VPN_net, avg_valid_loss)
        if earlystopping.early_stop:
            print("Early Stopping!")
            break

        # learning schdule
        scheduler.step()

        # output csvfile
        df = pd.DataFrame({'epoch': epoch_list, 'loss': train_loss_list, 'val_loss': valid_loss_list})
        df.to_csv("{0}/VPN_loss_{1}.csv".format(csvfile_savedir, date), index=None)
    

if __name__ == '__main__':
    main()


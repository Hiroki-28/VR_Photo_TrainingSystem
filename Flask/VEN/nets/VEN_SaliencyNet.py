import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import cv2
import numpy as np
import sys
import torch.nn.functional as F


class FullVggCompositionNet(nn.Module):
    def __init__(self):
        super(FullVggCompositionNet, self).__init__()

        model = models.vgg16(pretrained=False)
        self.features = model.features

        self.classifier = nn.Sequential(
            nn.Linear(7*7*512, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512,1)
	)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG16(nn.Module):
    def __init__(self, init_weights: bool=True):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.ReLU(True),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
    
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(True),
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
    
            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(True),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(True),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(256,512,3,1,1),
            nn.ReLU(True),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(True),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(True),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(True),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2)
        )

        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(7*7*512, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512,1)
        )

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #img_np = img.squeeze().cpu().numpy().transpose(1,2,0) #tensor -> numpyに変換
        #saliency_img = detect_saliency(img_np)
	#入力画像とそのSaliency_Mapの結合 (配列の結合)
        #x = torch.cat((img, saliency_img), 1).float() #(1,4,224,224)
        #print("x.shape:", x.shape)
        x = self.features(x)
        #x = x.view(x.size(0), -1)
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG16_4(nn.Module):

    def __init__(self):
        super(VGG16_4, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
	    nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
	    nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
	    nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
	    nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
	    nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
	    nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
	    nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
	    nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
	    nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
	    nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
	    nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
	    nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
	    nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.classifier = nn.Sequential(
            nn.Linear(7*7*512, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1),
        )


    def forward(self, x):

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    test = torch.randn([1,4,224,224])
    test = Variable(test)
    VGG16_4 = VGG16_4()
    VGG16 = VGG16()
    FullVggCompositionNet = FullVggCompositionNet()
    #output = VGG16_4(test)
    #output2 = VGG16(test)
    #output3 = FullVggCompositionNet(test)
    print("VGG16_4:{0}".format(VGG16_4))
    print("VGG16:{0}".format(VGG16))
    print("FullVggCompositionNet:{0}".format(FullVggCompositionNet))
    #print("output:{0}".format(output))
    print("DEBUG")

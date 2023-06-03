import torch, sys
import torch.nn as nn
import torch.nn.functional as F


#この論文で用いられていたResNet
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes)
            )
        
        # init weight (activation = relu -> kaiming_normal)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GatedBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(GatedBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False) # 32,128,16,16
        self.conv2 = nn.Conv2d(in_channels=in_planes, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)

        # init weight (activation = tanh, sigmoid -> xavier_normal)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=1.0)

    def forward(self, x):
        out = self.conv1(x)
        #split_size = int(list(out.size())[1]/2)
        out_1 = out.split(int(list(out.size())[1]/2), dim=1)[0]   # int(list(out.size())[1]/2) -> 64 or 128
        out_2 = out.split(int(list(out.size())[1]/2), dim=1)[1]
        out_1 = torch.tanh(self.conv2(out_1))
        out_2 = torch.sigmoid(self.conv2(out_2))
        out = torch.mul(out_1, out_2) # element-wise multi
        return out


class ResNet_GatedCNN(nn.Module):
    def __init__(self):
        super(ResNet_GatedCNN, self).__init__()
        # resnet
        self.block_1 = nn.Sequential(
                BasicBlock(3,64,1),
                BasicBlock(64,64,1))
        # resnet
        self.block_2 = nn.Sequential(
                BasicBlock(64,128,2),
                BasicBlock(128,128,1))
        # gated_cnn
        self.block_3 = nn.Sequential(
                GatedBlock(64,128),  #32,64,16,16 -> 32,128,16,16
                nn.MaxPool2d((2,2)), #32,128,16,16 -> 32,128,8,8
                GatedBlock(128,256))  
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()

    def forward(self, x, gated=False):
        out = self.block_1(x)
        if gated == False:
            out = self.block_2(out)
        elif gated == True:
            out = self.block_3(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        return out


class ColorHarmonyNet_w_labels(nn.Module):
    def __init__(self, color_net):
    #def __init__(self, color_net_center, color_net_neighbor):
        super(ColorHarmonyNet_w_labels, self).__init__()
        #特徴量を抽出するモデルを指定
        self.color_net = color_net
        
        self.label_info_block = nn.Sequential(
                nn.Linear(8,32),
                nn.ReLU(inplace=True),
                nn.Linear(32,64),
                nn.ReLU(inplace=True))

        #色の観点で美的である確率、隣接色と調和が取れている確率
        self.good_prob_c = nn.Sequential(
                nn.Linear(192,256),
                nn.ReLU(inplace=True),
                nn.Linear(256,1),
                nn.Sigmoid())
    
        self.good_prob_n = nn.Sequential(
                nn.Linear(704,256),
                nn.ReLU(inplace=True),
                nn.Linear(256,1),
                nn.Sigmoid())


    def forward(self, c, n1, n2, n3, n4, label): # label -> onehot
        #中心パッチの特徴
        feature_c = self.color_net(c, False)
        #隣接パッチの特徴
        feature_n1 = self.color_net(n1, True)
        feature_n2 = self.color_net(n2, True)
        feature_n3 = self.color_net(n3, True)
        feature_n4 = self.color_net(n4, True)
        #ラベル情報の特徴
        feature_label = self.label_info_block(label)
        feature_c = torch.cat((feature_label, feature_c), 1) # 64+128->192dim
        #concatenate
        feature_all = torch.cat((feature_c, feature_n1, feature_n2, feature_n3, feature_n4), 1)
        P_c = self.good_prob_c(feature_c)
        P_n = self.good_prob_n(feature_all)
        probability = 0.1*P_c + 0.9*P_n

        return probability

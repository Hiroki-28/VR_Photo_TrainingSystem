import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class SSD_320(nn.Module):
    def __init__(self, num_classes: int = 500, init_weights: bool = True):
        super(SSD_320, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # block1 (320×320)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),

            # block2 (160×160)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),

            # block3 (80×80)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),

            # block4 (40×40)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),

            # block5 (20×20)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=1, padding=1),

            # block6 (20×20)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3,3), dilation=6, padding=6),
            nn.ReLU(True),
            # block7 (20×20)
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(1,1), stride=(1,1), padding=0),
            nn.ReLU(True),
            # block8 (20×20)
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=0),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(2,2), padding=1), # better padding = 0
            nn.ReLU(True),
            nn.Dropout(0.5),

            # prediction_part (10×10)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(9,9), stride=(1,1), padding=0),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(1*1*1024, num_classes)
        )
        #self.avgpool = nn.AdaptiveAvgPool2d((1,1))

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
        x = self.features(x)
#        print("x.shape:", x.shape)
#        x = self.avgpool(x)
#        print("x.shape_after:", x.shape)
#        x = torch.flatten(x, 1)
#        x = nn.Linear(1024, self.num_classes)(x)
        return x


if __name__ == '__main__':
    test = torch.randn([1,3,320,320])
    ssd = SSD_320()
    print("test:", test)
    output = ssd(test)
    print("output:", output)


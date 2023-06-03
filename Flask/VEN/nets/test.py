import torch
from torch.autograd import Variable
import cv2
import numpy as np

def detect_saliency(img, scale=6, q_value=0.95, target_size=(224,224)):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     cv2.imwrite('test.jpg', img_gray)

#     W, H = img_gray.shape
#     img_resize = cv2.resize(img_gray, (H // scale, W // scale), interpolation=cv2.INTER_AREA)

    H, W = img_gray.shape
    img_resize = cv2.resize(img_gray, (W // scale, H // scale), interpolation=cv2.INTER_AREA)
    myFFT = np.fft.fft2(img_resize)
    myPhase = np.angle(myFFT)
    myLogAmplitude = np.log(np.abs(myFFT) + 0.000001)
    #A(f). 局所平均フィルタhn(f)->(3,3)
    myAvg = cv2.blur(myLogAmplitude, (3, 3))
    mySpectralResidual = myLogAmplitude - myAvg

    m = np.exp(mySpectralResidual) * (np.cos(myPhase) + complex(1j) * np.sin(myPhase))
    #逆フーリエ変換 np.fft.ifft2()
    saliencyMap = np.abs(np.fft.ifft2(m)) ** 2
    saliencyMap = cv2.GaussianBlur(saliencyMap, (9, 9), 2.5)
    saliencyMap = cv2.resize(saliencyMap, target_size, interpolation=cv2.INTER_LINEAR)
    threshold = np.quantile(saliencyMap.reshape(-1), q_value)
    if threshold > 0:
        saliencyMap[saliencyMap > threshold] = threshold
        saliencyMap = (saliencyMap - saliencyMap.min()) / threshold
    return saliencyMap

def main():
    test = torch.randn([1,3,224,224])
    test = Variable(test)
    test = test.squeeze().cpu().numpy().transpose(1,2,0)
    print(test.shape)
    sal_test = detect_saliency(test)

if __name__ == '__main__':
    main()

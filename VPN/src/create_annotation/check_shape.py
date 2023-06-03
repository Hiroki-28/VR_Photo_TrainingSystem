import glob, os, sys
from PIL import Image
import numpy as np

def main():
    image_list = glob.glob('../dataset/AADB/*.jpg')
    print('len(image_list):', len(image_list))
    for img_idx, img_path in enumerate(image_list):
        img = Image.open(img_path)
        #print(img.getpixel((0,0)))
        img = np.array(img)
        if len(img.shape) != 3:
            print('{0}\t{1}'.format(os.path.basename(img_path), img.shape))


if __name__ == '__main__':
    main()


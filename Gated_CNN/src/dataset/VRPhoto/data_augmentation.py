from PIL import Image
import glob
import os


def main():
    folder_path = './data_811/train/'
    filepath_list = glob.glob(os.path.join(folder_path, '*/*.jpg'))
    for idx, img_path in enumerate(filepath_list):
        print("{0}/{1}".format(idx+1, len(filepath_list)))
        img = Image.open(img_path)
        
        # rotate image
        img_90  = img.rotate(90, expand=True)
        img_180 = img.rotate(180, expand=True)
        img_270 = img.rotate(270, expand=True)
        
        # save image
        index = img_path.split('.jpg')[0]
        img_90.save("{0}_90.jpg".format(index))
        img_180.save("{0}_180.jpg".format(index))
        img_270.save("{0}_270.jpg".format(index))


if __name__ == '__main__':
    main()


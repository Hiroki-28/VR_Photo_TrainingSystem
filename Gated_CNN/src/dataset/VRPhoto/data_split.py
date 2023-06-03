import os
import glob
import shutil
import random
random.seed(0)


def get_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path


def main():
    # set path
    bad_path = './images/bad'
    good_path = './images/good'
    output_dirpath = './data_811'

    # get imagefile list and shuffle
    filelist_b = glob.glob("{0}/*.jpg".format(bad_path))
    filelist_g = glob.glob("{0}/*.jpg".format(good_path))
    random.shuffle(filelist_b)
    random.shuffle(filelist_g)

    # split dataset
    bad_train_li = filelist_b[:int(len(filelist_b)*0.8)]
    bad_valid_li = filelist_b[int(len(filelist_b)*0.8):int(len(filelist_b)*0.9)]
    bad_test_li = filelist_b[int(len(filelist_b)*0.9):]
    good_train_li = filelist_g[:int(len(filelist_g)*0.8)]
    good_valid_li = filelist_g[int(len(filelist_g)*0.8):int(len(filelist_g)*0.9)]
    good_test_li = filelist_g[int(len(filelist_g)*0.9):]

    # create train data
    output_bad = get_dir("{0}/train/bad".format(output_dirpath))
    output_good = get_dir("{0}/train/good".format(output_dirpath))
    for img_path in bad_train_li:
        shutil.copy("{0}".format(img_path), output_bad)
    for img_path in good_train_li:
        shutil.copy("{0}".format(img_path), output_good)

    # create valid data
    output_bad = get_dir("{0}/valid/bad".format(output_dirpath))
    output_good = get_dir("{0}/valid/good".format(output_dirpath))
    for img_path in bad_valid_li:
        shutil.copy("{0}".format(img_path), output_bad)
    for img_path in good_valid_li:
        shutil.copy("{0}".format(img_path), output_good)

    # create test data
    output_bad = get_dir("{0}/test/bad".format(output_dirpath))
    output_good = get_dir("{0}/test/good".format(output_dirpath))
    for img_path in bad_test_li:
        shutil.copy("{0}".format(img_path), output_bad)
    for img_path in good_test_li:
        shutil.copy("{0}".format(img_path), output_good)


if __name__ == '__main__':
    main()

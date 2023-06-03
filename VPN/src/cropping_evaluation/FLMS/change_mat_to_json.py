import json
import scipy.io


def main():
    mat = scipy.io.loadmat("500_image_dataset.mat")
    annotation = {}
    for i in range(len(mat['img_gt'])):
        bboxes = []
        for bbox in mat['img_gt'][i][0][1]:
            new_bbox = []
            for v in bbox:
                new_bbox.append(int(v))
            bboxes.append(new_bbox)
        annotation[mat['img_gt'][i][0][0][0]] = {}
        annotation[mat['img_gt'][i][0][0][0]] = bboxes

    with open("500_image_dataset.json", 'w') as f:
        json.dump(annotation, f, indent=2)


if __name__ == "__main__":
    main()

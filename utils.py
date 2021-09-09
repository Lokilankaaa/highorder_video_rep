import cv2
import numpy as np
import os
import json


def read_img(filename):
    return cv2.imread(filename, cv2.COLOR_BGR2RGB)


def hist2color(img):
    def calc_hist(channel):
        hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
        return np.average(range(256), weights=hist.flatten()).astype(int)

    r, g, b = map(calc_hist, [0, 1, 2])
    return r, g, b


def mapping_from_imgs(directory, out_json, num_workers=1):
    imgs = os.listdir(directory)
    mappings = dict()
    for img in imgs:
        r, g, b = hist2color(read_img(os.path.join(directory, img)))
        mappings['{}-{}-{}'.format(r, g, b)] = os.path.join(directory, img)
    with open(out_json, 'w') as f:
        json.dump(mappings, f)


def mapping_from_video(video_filename, out_json, num_workers=1):
    cam = cv2.VideoCapture(video_filename)
    mappings = dict()
    num = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        r, g, b = hist2color(frame)
        mappings['{}-{}-{}'.format(r, g, b)] = os.path.join('tmp', '{}.jpg'.format(num))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join('tmp', '{}.jpg'.format(num)), frame)
        num += 1
    with open(out_json, 'w') as f:
        json.dump(mappings, f)


def find_closest(rgb, inverse_map):
    tmp = np.linalg.norm(inverse_map - rgb, axis=1)
    i = np.argmin(tmp)
    return '-'.join(inverse_map[i].astype(str).tolist())


if __name__ == '__main__':
    pass

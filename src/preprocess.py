from pathlib import Path
import cv2
import argparse
import os
from numpy.core.fromnumeric import nonzero
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib.pyplot as plt
from itertools import combinations
from kornia.contrib import FaceDetector, FaceDetectorResult
import kornia as K
import sklearn


parser = argparse.ArgumentParser(description='Pytorch framework for training and fine-tuning models for face recognition')

parser.add_argument("--source_path", default="", type=str, help='Root path of the source images(default is None)')
parser.add_argument("--target_path", default="", type=str, help='Root path to save target images(default is None)')
parser.add_argument("--size", default=128, type=int, help='Target resolution with which the processed images is saved(default value is 128)')
parser.add_argument("--num", default=2000, type=int, help='Number of images to be processed(default value is 2000)')


def facedetect(src, tgt, size, dcount):
    device = torch.device('cuda')
    for dir_num in range(dcount):
        for img_num in range(1,5,1):
            src_img = f'{src}/id_{dir_num}/{dir_num}_{img_num}.png'
            tgt_img = f'{tgt}/id_{dir_num}/{dir_num}_{img_num}.png'
            os.makedirs(f'{tgt}/id_{dir_num}', exist_ok=True)
            face_detection = FaceDetector().to(device, torch.float32)
            #print(src_img)
            #print(f'{tgt}/id_{dir_num}')
            img_bgr = cv2.imread(src_img)
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = K.utils.image_to_tensor(np.asarray(img)).to(device, torch.float32)
            img = K.geometry.transform.resize(img, (size, size))
            with torch.no_grad():
                dets = face_detection(img.unsqueeze(0))
            det = [FaceDetectorResult(o) for o in dets]
            if len(det) > 0:
                x1, y1 = det[0].xmin.int(), det[0].ymin.int()
                x2, y2 = det[0].xmax.int(), det[0].ymax.int()
                roi = img[..., y1:y2, x1:x2]
            else:
                print("Exception: Face Not Detected")
                roi = img.clone()
            if roi.squeeze(0).shape[1] == 0 or roi.squeeze(0).shape[2] == 0:
                img = K.geometry.transform.resize(img, (size, size))
                img_np = K.utils.tensor_to_image(img)
                Image.fromarray(img_np.astype('uint8'), 'RGB').save(tgt_img)
            else:
                roi = K.geometry.transform.resize(roi, (size, size))
                img_np = K.utils.tensor_to_image(roi)
                Image.fromarray(img_np.astype('uint8'), 'RGB').save(tgt_img)

def main():
    global args
    args = parser.parse_args()
    facedetect(args.source_path, args.target_path, args.size, args.num)


if __name__ == '__main__':
    main()

#python /workspace/fairDL/src/preprocess.py --source_path="/workspace/fairDL/data/synthface" --target_path="/workspace/fairDL/data/synthface_processed" --size=128 --num=2000
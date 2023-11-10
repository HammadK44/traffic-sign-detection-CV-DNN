import argparse
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import random
import os

def inference(model_path, images_path):
    
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)  # Loading the trained YOLOv5 model

    imgs = os.listdir(images_path)
    random_images = random.sample(imgs, 5)

    fig, axs = plt.subplots(15, 1, figsize=(20, 10 * 15))

    for i, imgname in enumerate(random_images):
        img = cv2.imread(os.path.join(images_path, imgname))
        results = model(img)
        axs[i].imshow(cv2.cvtColor(np.squeeze(results.render()), cv2.COLOR_BGR2RGB))
        axs[i].set_title(imgname)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run YOLOv5 Inference')
    parser.add_argument('--model_path', type=str, help='Path to the YOLOv5 model')
    parser.add_argument('--images_path', type=str, help='Path to the directory containing images')

    args = parser.parse_args()

    inference(args.model_path, args.images_path)

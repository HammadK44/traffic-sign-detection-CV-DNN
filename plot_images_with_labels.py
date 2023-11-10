import argparse
import os
import cv2
import pandas as pd
from random import sample
import matplotlib.pyplot as plt

def plot_random(image_folder, df, num_images=5):
    random_image_names = sample(df['image_name'].tolist(), num_images)

    for i, image_name in enumerate(random_image_names):
        image_path = os.path.join(image_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        rows_for_image = df[df['image_name'] == image_name]

        plt.figure(figsize=(16, 16))
        plt.imshow(img)
        plt.title(f"{image_name}")
        plt.axis('off')

        for _, row in rows_for_image.iterrows():
            bbox_list = eval(row['bbox'])
            label = row['sign_type']

            if not isinstance(bbox_list[0], list):  # If there's only one bounding box, convert it to a list
                bbox_list = [bbox_list]

            for bbox in bbox_list:
                rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                     linewidth=2, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)

                plt.text(bbox[0], bbox[1], f"{label}", color='r', verticalalignment='top')

        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Random Images with Bounding Boxes and Labels')
    parser.add_argument('--image_folder', type=str, help='Path to the folder containing images')
    parser.add_argument('--annotations_file', type=str, help='Path to the CSV file containing annotations')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to plot')

    args = parser.parse_args()

    # Load the dataset
    df = pd.read_csv(args.annotations_file)

    plot_random(args.image_folder, df, args.num_images)
# File for running all data preprocessing.

import os
import pandas as pd

import parse_txt_annotations
import create_yolo_txts

train_images_folder = ".data/Set1Part0/images"
train_annotation_file = ".data/Set1Part0/labels/annotations.txt"

test_images_folder = ".data/Set2Part0/images"
test_annotation_file = ".data/Set2Part0/labels/annotations.txt"

# Creating annotations csv files
train_annot_csv = parse_txt_annotations.parse_sign_annotations_export_to_csv(train_annotation_file)
test_annot_csv = parse_txt_annotations.parse_sign_annotations_export_to_csv(test_annotation_file)

# Reading the csv files as dfs and removing invalid bbox rows from the dfs
df = pd.read_csv(os.path.join(os.path.dirname(train_annotation_file), train_annot_csv))
invalid_bbox = df['bbox'].apply(lambda bbox: eval(bbox) == [-1, -1, -1, -1])
df = df[~invalid_bbox]
df = df.reset_index(drop=True)

df_test = pd.read_csv(os.path.join(os.path.dirname(test_annotation_file), test_annot_csv))
invalid_bbox = df_test['bbox'].apply(lambda bbox: eval(bbox) == [-1, -1, -1, -1])
df_test = df_test[~invalid_bbox]
df_test = df_test.reset_index(drop=True)

# Creating yolo txt files for all train and test images using the annotations
create_yolo_txts.create_yolo_files(train_images_folder, df, os.path.dirname(train_annotation_file))
create_yolo_txts.create_yolo_files(test_images_folder, df_test, os.path.dirname(test_annotation_file))
os.remove(train_annotation_file)
os.remove(test_annotation_file)
import torch
import cv2
import os
import json
from tqdm import tqdm
from pathlib import Path

# Convert Predictions to COCO Format JSON

img_folder = './data/Set2Part0/images'
img_files = os.listdir(img_folder)

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/trained_model/best.pt',force_reload=True)

predictions_coco = []
for img_name in tqdm(img_files, desc="Inference"):
    img_path = os.path.join(img_folder, img_name)
    img = cv2.imread(img_path)

    detections = model(img)

    raw_detections = detections.xyxy[0].cpu().numpy()

    coco_results = []
    for detection in raw_detections:
        coco_result = {
            'image_id': img_name.split('.')[0] ,  # Use the full image name as the image_id, without the extension (.jpg)
            'category_id': int(detection[5]) + 1,
            'bbox': detection[0:4].tolist(),
            'score': float(detection[4])
        }
        predictions_coco.append(coco_result)
        
coco_format_pred = 'pred.json'
with open(coco_format_pred, 'w') as fw:
    json.dump(predictions_coco, fw)
        

        
# Convert Annotations to COCO Format JSON
# Credit to https://github.com/PrabhjotKaurGosal/ObjectDetectionScripts/blob/main/CovertAnnotations_YOLO_to_COCO_format.ipynb

categories = ['PRIORITY_ROAD', 'PASS_EITHER_SIDE', 'PASS_RIGHT_SIDE', 'GIVE_WAY', '70_SIGN', 
                  '90_SIGN', 'OTHER', '80_SIGN', '50_SIGN', 'PEDESTRIAN_CROSSING', '60_SIGN', '30_SIGN', 'NO_PARKING',
                  'PASS_LEFT_SIDE', '110_SIGN', 'STOP', '100_SIGN', 'NO_STOPPING_NO_STANDING', 'URDBL', '120_SIGN']

new_categories = [{'id': j + 1, 'name': label, 'supercategory': label} for j, label in enumerate(categories)]

write_json_context = dict()
write_json_context['info'] = {'description': '', 'url': '', 'version': '', 'year': 2021, 'contributor': '', 'date_created': '2021-02-12 11:00:08.5'}
write_json_context['licenses'] = [{'id': 1, 'name': None, 'url': None}]
write_json_context['categories'] = new_categories
write_json_context['images'] = []
write_json_context['annotations'] = []

directory_labels = os.fsencode("/kaggle/working/data/test/labels")
directory_images = os.fsencode("/kaggle/working/data/test/images")

file_number = 1
num_bboxes = 1

for file in os.listdir(directory_images):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        img_path = os.path.join(directory_images.decode("utf-8"), filename)
        base = os.path.basename(img_path)
        file_name_without_ext = os.path.splitext(base)[0]  # name of the file without the extension
        yolo_annotation_path = os.path.join(directory_labels.decode("utf-8"), file_name_without_ext + "." + 'txt')
        img_name = os.path.basename(img_path)  # name of the file without the extension
        img_context = {}
        height, width = cv2.imread(img_path).shape[:2]
        img_context['file_name'] = img_name
        img_context['height'] = height
        img_context['width'] = width
        img_context['date_captured'] = '2021-02-12 11:00:08.5'
        img_context['id'] = file_name_without_ext  # use file name without extension as image id
        img_context['license'] = 1
        img_context['coco_url'] = ''
        img_context['flickr_url'] = ''
        write_json_context['images'].append(img_context)

        try:
            with open(yolo_annotation_path, 'r') as f2:
                lines2 = f2.readlines()
        except:
            continue

        for i, line in enumerate(lines2):
            line = line.split(' ')
            bbox_dict = {}
            class_id, x_yolo, y_yolo, width_yolo, height_yolo = line[0:]
            x_yolo, y_yolo, width_yolo, height_yolo, class_id = (
                float(x_yolo),
                float(y_yolo),
                float(width_yolo),
                float(height_yolo),
                int(class_id),
            )
            bbox_dict['id'] = num_bboxes
            bbox_dict['image_id'] = file_name_without_ext
            bbox_dict['category_id'] = class_id + 1
            bbox_dict['iscrowd'] = 0
            h, w = abs(height_yolo * height), abs(width_yolo * width)
            bbox_dict['area'] = h * w
            x_coco = round(x_yolo * width - (w / 2))
            y_coco = round(y_yolo * height - (h / 2))
            if x_coco < 0:
                x_coco = 1
            if y_coco < 0:
                y_coco = 1
            bbox_dict['bbox'] = [x_coco, y_coco, w, h]
            bbox_dict['segmentation'] = [
                [x_coco, y_coco, x_coco + w, y_coco, x_coco + w, y_coco + h, x_coco, y_coco + h]
            ]
            write_json_context['annotations'].append(bbox_dict)
            num_bboxes += 1

        file_number = file_number + 1
        continue
    else:
        continue

coco_format_save_path = 'gt.json'
with open(coco_format_save_path, 'w') as fw:
    json.dump(write_json_context, fw)
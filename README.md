# Swedish Traffic Signs Detection using YOLOv5

This project focuses on detecting and classifying traffic signs in the **Swedish Traffic Sign Dataset (STSD)** using the YOLOv5 object detection model. This pipeline involves data processing, YOLO format conversion, model training, and inference.

## Dataset

The dataset used in this project consists of images of Swedish traffic signs along with corresponding annotations. Annotations are originally provided in a single .txt file, which includes the image names and their corresponding bounding box coordinates, sign type, sign status, sign size, sign centre, and sign aspect ratio.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/HammadK44/traffic-sign-detection-CV-DNN.git
cd traffic-sign-detection-CV-DNN
```

Clone the YOLOv5 repository into the '*traffic-sign-detection-CV-DNN*' repo.

```bash
git clone https://github.com/ultralytics/yolov5
```

### 2. Install YOLOv5 Dependencies 
```bash
pip install -r yolov5/requirements.txt
```

### 3. Download and setup the STSD Dataset and Annotations (Set1Part0 for Training, Set2Part0 for Testing)

```bash
mkdir data/Set1Part0/images
wget -P data/Set1Part0/images http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set1/Set1Part0.zip
unzip data/Set1Part0/images/Set1Part0.zip
rm data/Set1Part0/images/Set1Part0.zip
mkdir data/Set1Part0/labels
wget -P data/Set1Part0/labels http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set1/annotations.txt
```
```bash
mkdir data/Set2Part0/images
wget -P data/Set2Part0/images http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set2/Set2Part0.zip
unzip data/Set2Part0/images/Set2Part0.zip
rm data/Set2Part0/images/Set2Part0.zip
mkdir data/Set2Part0/labels
wget -P data/Set2Part0/labels http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set2/annotations.txt
```

### 4. Copy STSD VOC.yaml into YOLOv5 folder

```bash
cp ./VOC.yaml ./yolov5/VOC.yaml
```


## Running the Code

### 1. Data Processing

The data processing file ('*data_processing.py*') performs the following tasks:

- Reads the dataset annotations .txt file and converts into a csv file using the '*parse_txt_annotations.py*' script.
- Reads the annotation csv and filters out invalid bounding boxes ( [-1,-1,-1,-1] ).
- Creates YOLO format text files for training using the '*create_yolo_txts.py*' script.

Run the data processing file as follows:
```bash
python data_preprocessing.py
```

### 2. YOLOv5 Training

Running the YOLOv5 training script using the following command trains the model using the processed data. The training configuration is specified in '*VOC.yaml*'. This training was ran for a 100 epochs using the following command, without any modifications such as Data Augmentations, IoU threshold change, etc:

```bash
python ./yolov5/train.py --img 640 --batch 16 --epochs 100 --data ./yolov5/VOC.yaml --weights yolov5s.pt --workers 2
```

## 3. Inference

The inference script ('*inference.py*') uses the trained YOLOv5 model to make predictions on a set of random images. The results are displayed for visualization.

Run the inference script:

```bash
python inference.py --model_path "path-to-the-trained-model" --images_path "path-to-the-images"
```

## Results

Randomly selected images with model predictions are visualized using the *'inference.py'* output.

### 1. Train Dataset Evaluation Metrics

Following are the evaluation metrics using the training dataset, as well as the Confusion Matrix and Precision-Recall curves for all data classes:

![Training Metrics](./results/results_train_metrics.png)

| Class                    | Images | Instances |   P    |   R    | mAP50  |
|--------------------------|--------|-----------|-------|-------|--------|
| all                      | 1970   | 3169      | 0.908 | 0.849 | 0.886  | 0.647  |
| PRIORITY_ROAD            | 1970   | 470       | 0.962 | 0.904 | 0.934  | 0.694  |
| PASS_EITHER_SIDE         | 1970   | 31        | 0.973 | 0.935 | 0.936  | 0.757  |
| PASS_RIGHT_SIDE          | 1970   | 351       | 0.957 | 0.894 | 0.933  | 0.652  |
| GIVE_WAY                 | 1970   | 261       | 0.974 | 0.878 | 0.958  | 0.635  |
| 70_SIGN                  | 1970   | 255       | 0.966 | 0.996 | 0.99   | 0.771  |
| 90_SIGN                  | 1970   | 64        | 0.969 | 0.988 | 0.994  | 0.755  |
| OTHER                    | 1970   | 543       | 0.955 | 0.902 | 0.949  | 0.694  |
| 80_SIGN                  | 1970   | 106       | 0.8   | 0.915 | 0.897  | 0.686  |
| 50_SIGN                  | 1970   | 223       | 0.951 | 0.966 | 0.983  | 0.686  |
| PEDESTRIAN_CROSSING      | 1970   | 337       | 0.937 | 0.896 | 0.937  | 0.65   |
| 60_SIGN                  | 1970   | 48        | 0.00274 | 0.000114 | 0.329 | 0.249 |
| 30_SIGN                  | 1970   | 45        | 1     | 0.849 | 0.949  | 0.659  |
| NO_PARKING               | 1970   | 39        | 0.9   | 1     | 0.995  | 0.784  |
| PASS_LEFT_SIDE           | 1970   | 19        | 0.99  | 1     | 0.995  | 0.77   |
| 110_SIGN                 | 1970   | 98        | 0.948 | 0.924 | 0.962  | 0.673  |
| STOP                     | 1970   | 21        | 0.95  | 1     | 0.995  | 0.617  |
| 100_SIGN                 | 1970   | 77        | 0.981 | 0.987 | 0.994  | 0.797  |
| NO_STOPPING_NO_STANDING  | 1970   | 77        | 0.949 | 0.948 | 0.964  | 0.707  |
| URDBL                    | 1970   | 12        | 1     | 0     | 0.0274 | 0.0138 |
| 120_SIGN                 | 1970   | 92        | 1     | 0.998 | 0.995  | 0.699  |

![Confusion Matrix](./results/results_train_metrics.png)
![PR-Curve](./results/results_train_metrics.png)


### 2. Inference Results on Train Dataset
![Inference Result 1](./results/results1.png)
![Inference Result 2](./results/results2.png)
![Inference Result 3](./results/results3.png)
![Inference Result 4](./results/results4.png)

### 3. Test Dataset Evaluation Metrics

![Training Metrics](./results/results_train_metrics.png)

| Class                    | Images | Instances |   P    |   R    | mAP50  |
|--------------------------|--------|-----------|-------|-------|--------|
| all                      | 1807   | 3482      | 0.496 | 0.482 | 0.464  | 0.301  |
| PRIORITY_ROAD            | 1807   | 652       | 0.668 | 0.702 | 0.742  | 0.458  |
| PASS_EITHER_SIDE         | 1807   | 15        | 0.251 | 0.467 | 0.237  | 0.198  |
| PASS_RIGHT_SIDE          | 1807   | 500       | 0.562 | 0.658 | 0.683  | 0.459  |
| GIVE_WAY                 | 1807   | 121       | 0.555 | 0.76  | 0.773  | 0.457  |
| 70_SIGN                  | 1807   | 215       | 0.441 | 0.567 | 0.557  | 0.385  |
| OTHER                    | 1807   | 273       | 0.46  | 0.418 | 0.423  | 0.275  |
| 80_SIGN                  | 1807   | 193       | 0.338 | 0.316 | 0.351  | 0.241  |
| 50_SIGN                  | 1807   | 193       | 0.553 | 0.363 | 0.385  | 0.264  |
| PEDESTRIAN_CROSSING      | 1807   | 928       | 0.728 | 0.613 | 0.661  | 0.444  |
| 30_SIGN                  | 1807   | 9         | 0.214 | 0.333 | 0.164  | 0.0566 |
| NO_PARKING               | 1807   | 158       | 0.731 | 0.411 | 0.504  | 0.297  |
| STOP                     | 1807   | 57        | 0.665 | 0.351 | 0.413  | 0.257  |
| 100_SIGN                 | 1807   | 108       | 0.408 | 0.467 | 0.398  | 0.295  |
| NO_STOPPING_NO_STANDING  | 1807   | 60        | 0.371 | 0.315 | 0.207  | 0.134  |

Following are the evaluation metrics using the testing dataset. As can be seen, the results on using the test dataset are not as good as when using the train dataset. This means that the model may be overfitting to the training data, and its performance on unseen data, represented by the test dataset, is not as robust. Several factors could contribute to this discrepancy, such as the diversity of the test set, differences in lighting conditions, or variations in traffic sign poses.

## Training 2nd Run

In light of this, training was performed a second time, by including data augmentations in the '*./yolov5/data/hyps/hyp.heavy.2.yaml*' file. Following augmentations were made:

```bash
lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.2  # final OneCycleLR learning rate (lr0 * lrf) - Slightly higher value for more gradual decrease
momentum: 0.9  # SGD momentum/Adam beta1 - Increased momentum for better convergence
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 5.0  # warmup epochs (fractions ok) - Increased warm-up for better initialization
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.5  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 1.0  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.3  # IoU training threshold - Slightly higher value for more confident detections
anchor_t: 3.0  # anchor-multiple threshold - Lower value for better adaptation to object sizes
fl_gamma: 2.0  # focal loss gamma (efficientDet default gamma=1.5) - Increased gamma for more focus on hard examples
hsv_h: 0.03  # image HSV-Hue augmentation (fraction) - Slightly higher value for more color variations
hsv_s: 0.6  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.5  # image HSV-Value augmentation (fraction)
degrees: 5.0  # image rotation (+/- deg) - Small rotation for better generalization
translate: 0.2  # image translation (+/- fraction) - Increased translation for more positional variations
scale: 0.7  # image scale (+/- gain) - Higher scale for better adaptability to different object sizes
shear: 2.0  # image shear (+/- deg) - Slightly higher shear for more affine transformations
perspective: 0.002  # image perspective (+/- fraction), range 0-0.001 - Small perspective change for more realistic views
flipud: 0.5  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 1.0  # image mosaic (probability)
mixup: 0.5  # image mixup (probability)
copy_paste: 0.0  # segment copy-paste (probability)
```

Then trained:

```bash
python ./yolov5/train.py --img 640 --batch 16 --epochs 75 --data ./yolov5/VOC.yaml --weights yolov5s.pt --workers 2 --hyp ./yolov5/data/hyps/hyp.heavy.2.yaml
```

## Trained Model

The trained model can be used directly for inference. Model file can be found in the specified *model_path*, and the following script can be used to run the inference. 

```bash
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import random
import os 

model_path = './trained_model/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

imgs = os.listdir('./data/Set1Part0/images')
random_images = random.sample(imgs, 5)

fig, axs = plt.subplots(15, 1, figsize=(20, 10 * 15))

for i, imgname in enumerate(random_images):
    img = cv2.imread(f'/kaggle/working/data/images/{imgname}')
    results = model(img)
    axs[i].imshow(cv2.cvtColor(np.squeeze(results.render()), cv2.COLOR_BGR2RGB))
    axs[i].set_title(imgname)
    
plt.show()
```

## License

This project is licensed under the MIT License

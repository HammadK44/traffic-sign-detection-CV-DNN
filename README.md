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

The data processing file ('*data_preprocessing.py*') performs the following tasks:

- Reads the dataset annotations .txt file and converts into a csv file using the '*parse_txt_annotations.py*' script.
- Reads the annotation csv and filters out invalid bounding boxes ( [-1,-1,-1,-1] ).
- Creates YOLO format text files for training using the '*create_yolo_txts.py*' script.

Run the data processing file as follows:
```bash
python data_preprocessing.py
```

### 2. YOLOv5 Training

Running the YOLOv5 training script using the following command trains the model using the processed data. The training configuration is specified in '*VOC.yaml*'. I ran the training for a 100 epochs using the following command:

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

Randomly selected images with model predictions are visualized in the *'inference.py'* output.

### Training Metrics
![Training Metrics](./results/results_train_metrics.png)

### Inference Results
![Inference Result 1](./results/results1.png)
![Inference Result 2](./results/results2.png)
![Inference Result 3](./results/results3.png)
![Inference Result 4](./results/results4.png)


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

import os
import cv2

def create_yolo_files(image_folder, df, output_dir):
    
    label_mapping = {
        'PRIORITY_ROAD': 0,
        'PASS_EITHER_SIDE': 1,
        'PASS_RIGHT_SIDE': 2,
        'GIVE_WAY': 3,
        '70_SIGN': 4,
        '90_SIGN': 5,
        'OTHER': 6,
        '80_SIGN': 7,
        '50_SIGN': 8,
        'PEDESTRIAN_CROSSING': 9,
        '60_SIGN': 10,
        '30_SIGN': 11,
        'NO_PARKING': 12,
        'PASS_LEFT_SIDE': 13,
        '110_SIGN': 14,
        'STOP': 15,
        '100_SIGN': 16,
        'NO_STOPPING_NO_STANDING': 17,
        'URDBL': 18,
        '120_SIGN': 19
    }
    
    for index, row in df.iterrows():
        image_name = row['image_name']
        image_path = os.path.join(image_folder, image_name)
        img = cv2.imread(image_path)
        img_height, img_width, _ = img.shape

        yolo_filename = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".txt")
        
        with open(yolo_filename, "w") as yolo_file:
            rows_for_image = df[df['image_name'] == image_name]

            for _, row in rows_for_image.iterrows():
                bbox_list = eval(row['bbox'])
                label = row['sign_type']
                label_id = label_mapping[label]

                # YOLO format: <class_id> <center_x> <center_y> <width> <height>
                x_center = (bbox_list[0] + bbox_list[2]) / (2 * img_width)
                y_center = (bbox_list[1] + bbox_list[3]) / (2 * img_height)
                width = (bbox_list[2] - bbox_list[0]) / img_width
                height = (bbox_list[3] - bbox_list[1]) / img_height

                yolo_file.write(f"{label_id} {x_center} {y_center} {width} {height}\n")
import os 
import xml.etree.ElementTree as ET
import cv2

def create_xml_file(image_folder, df, output_folder = "./data/labels"):
    os.makedirs(output_folder, exist_ok=True)

    for _, row in df.iterrows():
        image_name = row['image_name']
        width, height, _ = cv2.imread(os.path.join(image_folder, image_name)).shape
        xml_filename = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}.xml")

        root = ET.Element("annotation")

        ET.SubElement(root, "folder").text = "images"
        ET.SubElement(root, "filename").text = image_name
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"  # Assuming it's a 3-channel image

        ET.SubElement(root, "segmented").text = "0"

        rows_for_image = df[df['image_name'] == image_name]

        for _, row_for_bbox in rows_for_image.iterrows():
            bbox = eval(row_for_bbox['bbox'])
            label = row_for_bbox['sign_type']
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = label
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "occluded").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(bbox[0]))
            ET.SubElement(bndbox, "ymin").text = str(int(bbox[1]))
            ET.SubElement(bndbox, "xmax").text = str(int(bbox[2]))
            ET.SubElement(bndbox, "ymax").text = str(int(bbox[3]))

        tree = ET.ElementTree(root)
        tree.write(xml_filename)


def convert_box(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

def convert_voc_to_yolo():
    for anno in os.listdir('./data/labels'):
        if anno.split('.')[1] == 'xml':
            file_name = anno.split('.')[0]
            out_file = open(f'./data/labels/{file_name}.txt', 'w')

            tree = ET.parse(os.path.join('./data/','labels', anno))
            root = tree.getroot()
            size = root.find('size')        
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            names = ['PRIORITY_ROAD', 'PASS_EITHER_SIDE', 'PASS_RIGHT_SIDE', 'GIVE_WAY', '70_SIGN', 
                     '90_SIGN', 'OTHER', '80_SIGN', '50_SIGN', 'PEDESTRIAN_CROSSING', '60_SIGN', '30_SIGN', 'NO_PARKING',
                     'PASS_LEFT_SIDE', '110_SIGN', 'STOP', '100_SIGN', 'NO_STOPPING_NO_STANDING', 'URDBL', '120_SIGN']

            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls in names and int(obj.find('difficult').text) != 1:
                    xmlbox = obj.find('bndbox')
                    bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
                    cls_id = names.index(cls)  # class id
                    out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')
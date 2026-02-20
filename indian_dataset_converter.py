import argparse
import os
import xml.etree.ElementTree as ET
import cv2
import shutil
from tqdm import tqdm

from utils import convert_bbox


def init():
    parser = argparse.ArgumentParser(description="Convert Indian dataset to YOLO format.")
    parser.add_argument('--root_dataset_path',
                        type=str,
                        required=True,
                        help="Path to image of the Indian dataset.")

    parser.add_argument('--output_path',
                        type=str,
                        required=True,
                        help="Path of the output YOLO formatted dataset.")

    return parser.parse_args()


def run(root_dataset_path: str, 
        output_path: str):

    images_out = os.path.join(output_path, "images")
    labels_out = os.path.join(output_path, "labels")
    cropped_out = os.path.join(output_path, "cropped_plates")

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    os.makedirs(cropped_out, exist_ok=True)

    recognition_file = os.path.join(output_path, "recognition_labels.txt")

    # Collect all xml files recursively
    xml_files = []
    for root, _, files in os.walk(root_dataset_path):
        for file in files:
            if file.endswith(".xml"):
                xml_files.append(os.path.join(root, file))

    print(f"Found {len(xml_files)} XML files.")

    with open(recognition_file, "w", encoding="utf-8") as rec_file:

        for xml_full_path in tqdm(xml_files):

            tree = ET.parse(xml_full_path)
            root = tree.getroot()

            filename = root.find("filename").text
            folder_of_xml = os.path.dirname(xml_full_path)

            image_full_path = os.path.join(folder_of_xml, filename)

            if not os.path.exists(image_full_path):
                print(f"Image not found: {image_full_path}")
                continue

            image = cv2.imread(image_full_path)
            if image is None:
                print(f"Failed to read image: {image_full_path}")
                continue

            height_img, width_img = image.shape[:2]

            base_name = os.path.splitext(filename)[0]

            # Avoid overwriting if same name appears in different folders
            unique_name = base_name + "_" + str(abs(hash(xml_full_path)) % 100000)

            # Copy image
            new_image_name = unique_name + ".jpg"
            shutil.copy(image_full_path, os.path.join(images_out, new_image_name))

            yolo_label_path = os.path.join(labels_out, unique_name + ".txt")

            with open(yolo_label_path, "w") as yolo_file:

                for obj in root.findall("object"):

                    plate_text = obj.find("name").text.strip()

                    xml_box = obj.find("bndbox")
                    xmin = int(xml_box.find("xmin").text)
                    ymin = int(xml_box.find("ymin").text)
                    xmax = int(xml_box.find("xmax").text)
                    ymax = int(xml_box.find("ymax").text)

                    # YOLO detection label
                    bbox = convert_bbox(
                        (width_img, height_img),
                        (xmin, ymin, xmax, ymax)
                    )

                    yolo_file.write(
                        f"0 {bbox[0]:.6f} {bbox[1]:.6f} "
                        f"{bbox[2]:.6f} {bbox[3]:.6f}\n"
                    )

                    # Crop plate
                    plate_crop = image[ymin:ymax, xmin:xmax]

                    crop_filename = unique_name + ".jpg"
                    crop_path = os.path.join(cropped_out, crop_filename)

                    cv2.imwrite(crop_path, plate_crop)

                    # Write recognition label
                    rec_file.write(f"{crop_filename}\t{plate_text}\n")

    print("✅ Conversion completed successfully!")
    print("Recognition labels saved at:", recognition_file)


if __name__ == "__main__":
    args = init()
    run(args.root_dataset_path, 
        args.output_path)
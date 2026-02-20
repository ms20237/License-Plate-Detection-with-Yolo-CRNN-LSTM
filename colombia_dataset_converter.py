import argparse
import os
import shutil
from tqdm import tqdm


def init():
    parser = argparse.ArgumentParser(description="Merge train/test cropped plates into single CRNN dataset.")
    parser.add_argument('--root_dataset_path',
                        type=str,
                        required=True,
                        help="Path to dataset root (contains train/ and test/).")

    parser.add_argument('--output_path',
                        type=str,
                        required=True,
                        help="Path to output dataset.")

    return parser.parse_args()


def run(root_dataset_path: str, 
        output_path: str):

    train_path = os.path.join(root_dataset_path, "train")
    test_path = os.path.join(root_dataset_path, "test")

    images_out = os.path.join(output_path, "images")
    os.makedirs(images_out, exist_ok=True)

    recognition_file = os.path.join(output_path, "recognition_labels.txt")

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

    all_images = []
    
    if os.path.exists(train_path):
        all_images += [(train_path, f) for f in os.listdir(train_path)
                       if f.lower().endswith(valid_ext)]

    if os.path.exists(test_path):
        all_images += [(test_path, f) for f in os.listdir(test_path)
                       if f.lower().endswith(valid_ext)]

    print(f"Total images found: {len(all_images)}")

    with open(recognition_file, "w", encoding="utf-8") as rec_file:

        for idx, (folder, img_name) in enumerate(tqdm(all_images)):

            label = os.path.splitext(img_name)[0].strip().upper()
            label = "".join(c for c in label if c.isalnum())

            if not label:
                continue

            src_path = os.path.join(folder, img_name)

            new_name = f"plate_{idx:06d}.jpg"
            dst_path = os.path.join(images_out, new_name)

            shutil.copy(src_path, dst_path)

            rec_file.write(f"{new_name}\t{label}\n")

    print("\n✅ All data merged successfully!")
    print("Images folder:", images_out)
    print("Label file:", recognition_file)


if __name__ == "__main__":
    args = init()
    run(args.root_dataset_path, args.output_path)
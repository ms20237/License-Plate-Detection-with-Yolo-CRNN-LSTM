import cv2
import os
import shutil
import numpy as np
import torch
import string
import torchvision.transforms as transforms
from collections import Counter
from torch.utils.data import Dataset


def preprocess_plate(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 32))
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32)
    img = img.unsqueeze(0).unsqueeze(0)                 # (1,1,H,W)
    return img


def ctc_decode(preds, alphabet):
    # preds: (B, T, C)
    preds = torch.argmax(preds, dim=2)  # (B, T)

    decoded = ""
    previous = -1

    for p in preds[0]:  # first sample in batch
        p = p.item()

        if p != 0 and p != previous:  # remove blanks & duplicates
            decoded += alphabet[p - 1]

        previous = p

    return decoded


def convert_bbox(size, box):
    width, height = size
    xmin, ymin, xmax, ymax = box

    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height

    return x_center, y_center, w, h


class PlateDataset(Dataset):
    def __init__(self, images_folder, labels_file):
        self.images_folder = images_folder
        self.samples = []

        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                img_name, text = line.strip().split("\t")
                self.samples.append((img_name, text))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 128)),              # H, W
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def encode_text(self, text):
        CHARS = string.digits + string.ascii_uppercase
        CHAR2IDX = {c: i+1 for i, c in enumerate(CHARS)}
        
        return [CHAR2IDX[c] for c in text if c in CHAR2IDX]

    def __getitem__(self, idx):
        img_name, text = self.samples[idx]

        img_path = os.path.join(self.images_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = self.transform(img)

        label = torch.tensor(self.encode_text(text), dtype=torch.long)

        return img, label, len(label)
    

def collate_fn(batch):
    imgs, labels, lengths = zip(*batch)

    imgs = torch.stack(imgs)

    labels_concat = torch.cat(labels)
    label_lengths = torch.tensor(lengths)

    return imgs, labels_concat, label_lengths


def decode_predictions(outputs, alphabet):
    """Greedy CTC decoder"""
    outputs = outputs.argmax(2)  # (B, T)
    decoded = []

    for seq in outputs:
        prev = -1
        text = ""
        for idx in seq:
            idx = idx.item()
            if idx != prev and idx != 0:
                text += alphabet[idx - 1]
            prev = idx
        decoded.append(text)

    return decoded


def cer(preds, targets):
    """Character Error Rate"""
    total_chars = 0
    total_errors = 0

    for p, t in zip(preds, targets):
        total_chars += len(t)
        total_errors += edit_distance(p, t)

    return total_errors / total_chars if total_chars > 0 else 0


def edit_distance(s1, s2):
    """Levenshtein distance"""
    dp = np.zeros((len(s1)+1, len(s2)+1))

    for i in range(len(s1)+1):
        dp[i][0] = i
    for j in range(len(s2)+1):
        dp[0][j] = j

    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )

    return dp[len(s1)][len(s2)]
    
    
def clean_and_analyze_labels(labels_file, min_char_freq=20):
    """
    - Converts labels to uppercase
    - Removes samples containing rare characters
    - Prints dataset statistics
    - Returns path to cleaned label file
    """

    print("\n🔎 Analyzing dataset...")

    # First pass: count characters
    char_counter = Counter()
    samples = []

    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                continue

            img_name, text = parts
            text = text.upper()

            samples.append((img_name, text))

            for c in text:
                char_counter[c] += 1

    print(f"Total samples before cleaning: {len(samples)}")
    print("\nCharacter frequency:")
    for k, v in sorted(char_counter.items()):
        print(f"{k}: {v}")

    # Detect rare characters
    rare_chars = {c for c, count in char_counter.items() if count < min_char_freq}

    if rare_chars:
        print("\n⚠ Removing samples containing rare characters:", rare_chars)

    # Second pass: remove samples with rare chars
    cleaned_samples = []
    for img_name, text in samples:
        if any(c in rare_chars for c in text):
            continue
        cleaned_samples.append((img_name, text))

    print(f"Total samples after cleaning: {len(cleaned_samples)}")

    # Write cleaned file
    cleaned_file = labels_file.replace(".txt", "_cleaned.txt")

    with open(cleaned_file, "w", encoding="utf-8") as f:
        for img_name, text in cleaned_samples:
            f.write(f"{img_name}\t{text}\n")

    print("✅ Cleaned label file saved at:", cleaned_file)

    return cleaned_file


def downsample_digits(labels_file, max_digit_ratio=0.6):
    """
    Reduces samples where digits dominate too heavily.
    """

    print("\n⚖ Applying digit balancing...")

    balanced_samples = []

    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            img_name, text = line.split("\t")

            digit_count = sum(c.isdigit() for c in text)
            ratio = digit_count / len(text)

            if ratio > max_digit_ratio:
                # randomly skip some digit-heavy samples
                if torch.rand(1).item() > 0.5:
                    continue

            balanced_samples.append((img_name, text))

    balanced_file = labels_file.replace(".txt", "_balanced.txt")

    with open(balanced_file, "w", encoding="utf-8") as f:
        for img_name, text in balanced_samples:
            f.write(f"{img_name}\t{text}\n")

    print("✅ Balanced label file saved at:", balanced_file)

    return balanced_file
    

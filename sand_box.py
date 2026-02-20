import cv2
import torch
from utils import preprocess_plate, ctc_decode
from CRNN import CRNN

device = "cuda" if torch.cuda.is_available() else "cpu"
alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
num_classes = len(alphabet) + 1

crnn = CRNN(num_classes).to(device)

# 🔥 LOAD TRAINED MODEL
crnn.load_state_dict(torch.load(r"", map_location=device))
crnn.eval()

img = cv2.imread(r"")

if img is None:
    print("Image failed to load")
    exit()

tensor = preprocess_plate(img).to(device)

with torch.no_grad():
    preds = crnn(tensor)

print("Decoded:", ctc_decode(preds, alphabet))

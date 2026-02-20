import cv2
import torch
from utils import preprocess_plate, ctc_decode
from CRNN import CRNN

device = "cuda" if torch.cuda.is_available() else "cpu"
alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
num_classes = len(alphabet) + 1

crnn = CRNN(num_classes).to(device)

# 🔥 LOAD TRAINED MODEL
crnn.load_state_dict(torch.load(r"D:\My_Heritage\license_plate_detection\models\recg_models\crnn_model.pth", map_location=device))
crnn.eval()

img = cv2.imread(r"D:\My_Heritage\license_plate_detection\dataset\Indian_vehicle_license_plate_dataset_yolo_format\images\00b42b2c-f193-4863-b92c-0245cbc816da___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_Nissan-Terrano-Petrol-Review-Images-Black-Front-Angle_80580.jpg")

if img is None:
    print("Image failed to load")
    exit()

tensor = preprocess_plate(img).to(device)

with torch.no_grad():
    preds = crnn(tensor)

print("Decoded:", ctc_decode(preds, alphabet))

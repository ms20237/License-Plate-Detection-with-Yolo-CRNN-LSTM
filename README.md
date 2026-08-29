# License-Plate-Detection-with-Yolo-CRNN-LSTM

This project implements a License Plate Recognition (LPR) system using a CRNN (Convolutional Recurrent Neural Network) and LSTM with CTC Loss for sequence prediction.
The model is trained to recognize full license plate text directly from cropped plate images.

## :ledger: Index
- [Dataset](#beginner-dataset)
- [Models](#beginner-models)
- [Repository Structure](#file_folder-repository-structure)
- [Installation](#electric_plug-installation)
- [Future Improvements](#construction-Future_Improvements)
- [License](#lock-license)


## :beginner: Dataset
We trained YOLO-v8s on this dataset [here](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e) and version [13](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/13) of it.

and for letter recognization we use 2 datasets: 
- [Indian vehicle license plate dataset](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset)
- [colombia dataset example](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/colombia_dataset_example.zip)


## :beginner: Models
We use Yolo-v8s model for plate detection and then crop plate part and make some preprocess and then use CRNN/LSTM for recognizing letters and numbers.


## :file_folder: Repository Structure
This repository contain several script for converting those 2 datasets to format which can train CRNN/LSTM model which are colombia_dataset_converter and indian_dataset_converter. and CRNN is structure of letter/number recognization model.


## :electric_plug: Installation
Use [conda](https://docs.conda.io/en/latest/)
 to create and manage the project environment, and [pip](https://pip.pypa.io/en/stable/)
 to install additional dependencies such as foobar.

- Clone the repository:
```bash
git clone https://github.com/ms20237/License-Plate-Detection-with-Yolo-CRNN-LSTM.git
cd License-Plate-Detection-with-Yolo-CRNN-LSTM
```
- Install dependencies:
```bash
pip install torch torchvision matplotlib tqdm ultralytics
```
- Or create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

for running convert script you have to do this:
```bash
python .\convert_indian_dataset_to_yolo_format.py --root_dataset_path  ".\dataset\Indian vehicle license plate dataset\State-wise_OLX"  --output_path ".\dataset\Indian_vehicle_license_plate_dataset_yolo_format"
```
change "--root_dataset_path" and "--output_path" to your real dataset and ourput path.

and also for training model run this in you command:
```bash
python .\train_lstm_CRNN_model.py  --images_folder ".\dataset\Indian_vehicle_license_plate_dataset_yolo_format\images"   --labels_file ".\dataset\Indian_vehicle_license_plate_dataset_yolo_format\recognition_labels.txt"  --epochs 80   --val_split 0.15  --test_split 0.15
```
change "--image_folder" and "--labels_file" to your real dataset and ourput path  and for changing number of epochs and split parts change these arguments: "--epochs", "--val_split", "--test_split"

## :construction: Future Improvements
- Beam Search decoding
- Transformer-based OCR
- Stronger augmentation
- Synthetic plate generation
- End-to-end detection + recognition integration
- Real-time inference optimization


## :lock: License
This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).


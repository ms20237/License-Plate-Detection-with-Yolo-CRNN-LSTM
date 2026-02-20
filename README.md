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


## :gear: Models
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
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## :construction: Future Improvements
- Beam Search decoding
- Transformer-based OCR
- Stronger augmentation
- Synthetic plate generation
- End-to-end detection + recognition integration
- Real-time inference optimization


## :lock: License
This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).


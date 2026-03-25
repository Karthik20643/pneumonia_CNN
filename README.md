# Pneumonia Detection from Chest X-rays using CNN (PyTorch)

## Project Overview
This project implements a Convolutional Neural Network (CNN) using PyTorch to detect Pneumonia from Chest X-ray images. The model is trained on a publicly available medical imaging dataset and is able to classify X-ray images into Normal and Pneumonia classes.

---

## Dataset
Dataset used: Chest X-ray Pneumonia Dataset (Kaggle)  
Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset contains three folders:
- Train
- Validation
- Test

Each folder contains two classes:
- NORMAL
- PNEUMONIA

---

## Technologies Used
- Python
- PyTorch
- Torchvision
- Matplotlib
- Scikit-learn
- Pillow

---

## Model Architecture
The Convolutional Neural Network (CNN) used in this project consists of:

1. Convolution Layer (Conv2D)
2. ReLU Activation Function
3. Max Pooling Layer
4. Convolution Layer (Conv2D)
5. ReLU Activation Function
6. Max Pooling Layer
7. Fully Connected Layer
8. Output Layer (2 classes: Normal, Pneumonia)

The CNN automatically extracts important features from X-ray images and uses them for classification.

---

## Training Details

| Parameter | Value |
|----------|------|
| Image Size | 224 x 224 |
| Batch Size | 32 |
| Epochs | 5 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Framework | PyTorch |

---

## Results

| Metric | Value |
|-------|------|
| Training Accuracy | 97.95% |
| Test Accuracy | 70.03% |
| Final Training Loss | 0.0553 |

### Training Loss Graph
<img width="1904" height="1073" alt="loss_graph" src="https://github.com/user-attachments/assets/3b1888cb-49ff-4e3d-afee-28a4c9a6d49e" />


### Training Accuracy Graph
<img width="1909" height="1076" alt="accuracy_graph" src="https://github.com/user-attachments/assets/5b924962-1246-4ada-b8e1-292d9a3bcd18" />


---

## Project Structure
<img width="433" height="600" alt="image" src="https://github.com/user-attachments/assets/e486f39b-6482-4674-90d3-4b14afa7b1f4" />



---

## How to Run the Project

### 1. Install Required Libraries
pip install torch torchvision matplotlib scikit-learn pillow

### 2. Train the Model
python train.py


---

## Conclusion
This project demonstrates how deep learning and Convolutional Neural Networks can be used in medical image analysis to detect diseases like Pneumonia from Chest X-ray images. The model was successfully trained and evaluated using PyTorch.

---

## Author
Karthik S S

# 📘 Fall Detection Project

This repository contains an automated **Fall Detection System** built using motion sensor data collected from a **wrist-worn device**. The system uses **Deep Learning (LSTM and future GRU models)** to classify human movements as either a **Fall** or **Activity of Daily Living (ADL)**.

---

## 📂 Data & Preprocessing

The model is trained on **DataSet.csv**, which contains continuous wrist-sensor measurements.

### **📑 Data Specification**

| Detail | Specification |
|--------|--------------|
| **Data Source** | DataSet.csv |
| **Sensor Type** | Wrist Motion Sensor |
| **Features** | 6 axes → 3 Accelerometer (g), 3 Angular Velocity (deg/s) |
| **Sequence Length** | 50 time steps per sample |
| **Fall Activity IDs** | 7, 8, 9, 10, 11 |
| **Imbalance Handling** | Class Weights applied |

### **🔧 Preprocessing Includes**
- Cleaning raw sensor data  
- Scaling all features  
- Creating 50-step sequences (sliding windows)  
- Correct labeling of Fall vs ADL  
- Applying class weights to handle Fall scarcity  

---

## 🧠 Deep Learning Models

### **1️⃣ LSTM Model (Trained & Saved)**

The LSTM network is designed to capture temporal patterns from sensor motion data.

**Architecture**
- LSTM Layer with 64 units  
- Dropout: 0.2  
- Dense Output Layer with Sigmoid activation  

**Performance (Test Set)**
- **Accuracy:** ~98.45%  
- **Precision (Fall):** ~0.9925  
- **Recall (Fall):** ~0.9838  

**Saved Model File:**  fall_detection_model.h5


---

### **2️⃣ GRU Model (Future Work)**

A lighter, faster alternative to the LSTM model.

**Planned Goals**
- Build a GRU model with 64 units  
- Compare its accuracy, recall, and training speed with the LSTM model  

---

## 🚀 Execution Guide

### **1. Prepare the Data**
Run:
```bash
python data_preprocessor.py
```
This cleans, scales, sequences, and labels the dataset.

### **2. Train and Save the Model**
Run:
```bash
python train_model.py
```
This:
- Trains the LSTM model
- Applies class weights for imbalance
- Saves the model as fall_detection_model.h5

### **3. Predict & Analyze**
Run:
```bash
python predict_fall.py
```
This:
- Loads the saved model.
- Makes predictions on the test set
- Prints
- - Classification Report
- - Confusion Matrix



---
## 📌 Summary

This project presents a robust and accurate approach to detecting falls using wrist-mounted sensor data and deep learning techniques. With a highly performing LSTM model already trained, the next step is to integrate and compare a GRU model for optimized performance.
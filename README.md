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

---

### **2️⃣ GRU Model (Future Work)**

A lighter, faster alternative to the LSTM model.



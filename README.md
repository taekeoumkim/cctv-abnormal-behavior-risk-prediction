# CCTV Abnormal Behavior Risk Prediction

This repository contains the implementation of a research project on  
**early risk prediction of abnormal behavior from CCTV videos** using deep learning.

The goal of this study is not to classify abnormal behavior types, but to  
**predict the risk of an abnormal event before it occurs**, based on human movement patterns observed in CCTV footage.

---

## 📌 Research Overview

- **Task**: Binary classification (Normal / Risk)
- **Objective**: Early detection of potentially dangerous situations
- **Input**: Time-series motion features extracted from CCTV videos
- **Model**: LSTM-based sequence model
- **Dataset**: AI Hub Abnormal Behavior CCTV Dataset

---

### File Descriptions

- **`preprocess.py`**  
  Extracts motion-based time-series features from CCTV videos and XML annotations.
  - Parses abnormal event start time from XML
  - Generates pre-event windows
  - Detects persons using YOLO
  - Extracts motion features (dx, dy, speed, acceleration)
  - Saves processed data as `.npy` files

- **`Predicting_abnormal.ipynb`**  
  Trains and evaluates an LSTM-based risk prediction model.
  - Scenario-level group split
  - Class imbalance handling
  - ROC-AUC / PR-AUC evaluation
  - Threshold-based analysis

- **`parameter_tuning.ipynb`**  
  Performs grid-based hyperparameter tuning for:
  - LSTM units
  - Dropout rate
  - Batch size
  - Class weight settings

---

## ⚙️ Experimental Setup

- **Window length**: 5 seconds  
- **Pre-event observation range**: 30 seconds  
- **Features per frame**:  
  - Δx, Δy
  - Speed
  - Acceleration (x, y)
- **Sequence length**: 149 timesteps
- **Classes**:
  - 0: Normal
  - 1: Risk (imminent abnormal behavior)

---

## 📊 Evaluation Metrics

The model is evaluated using the following metrics:

- **ROC-AUC**
- **PR-AUC**
- **Recall-focused threshold analysis**
- **Confusion matrix & classification report**

These metrics were chosen to reflect real-world CCTV surveillance requirements,  
where missing a risky situation is more critical than false alarms.

---

## 📁 Dataset

This project uses the **AI Hub Abnormal Behavior CCTV Dataset**.

⚠️ Due to data usage restrictions, the raw videos are **not included** in this repository.  
Only preprocessing and training code is provided.

---

## 🔁 Reproducibility

- All experiments were implemented in Python
- Random seeds were fixed for reproducibility
- The full preprocessing and training pipeline is included

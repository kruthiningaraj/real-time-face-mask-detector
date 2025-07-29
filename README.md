# 😷 Real-Time Face Mask Detector  
### 📊 Data Mining & Warehousing Course Project

This repository implements a **real-time face mask detection system** developed as part of the **Data Mining and Warehousing coursework**.

---

## 🧠 Features:
- Detects and classifies faces as **'With Mask'** or **'Without Mask'**.
- Built using **Python, Keras (TensorFlow backend), and OpenCV**.
- CNN for accurate classification with image preprocessing and augmentation.
- Real-time inference using OpenCV webcam feed.

---

## 📂 Dataset:
We use the **[Face Mask Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)**.

Organize as:
```
data/face_mask_dataset/
 ├── train/
 ├── val/
 └── test/
```

---

## 🚀 Getting Started:
1️⃣ Clone the repository:
```bash
git clone https://github.com/your-username/real-time-face-mask-detector.git
cd real-time-face-mask-detector
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

3️⃣ Train CNN:
```bash
python src/train_cnn.py
```

4️⃣ Evaluate model:
```bash
python src/evaluate.py
```

5️⃣ Run real-time detector:
```bash
python src/real_time_detector.py
```

Press **'q'** to exit webcam stream.

---

## 🏆 Highlights:
- Developed as part of **Data Mining and Warehousing course**.
- Real-time computer vision application with **low-latency detection**.
- End-to-end deep learning pipeline with deployment-ready OpenCV integration.

---

## 📜 License:
MIT License © 2025

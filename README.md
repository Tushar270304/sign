# 🖐️ Gujarati Hand Gesture Recognition

This project performs real-time hand gesture recognition for Gujarati sign language using a webcam, OpenCV, and a pre-trained deep learning model. It includes tools for data collection, model training, and live testing.

---

## 📁 Project Structure

```
.
├── Rasa-VariableFont_wght.ttf       # Font file (possibly for GUI or display)
├── datacollection.py                # Script to collect gesture data using webcam
├── datanew.py                       # Data preprocessing or model training script
├── new.py                           # Main script for live prediction and gesture recognition
├── test.py                          # Additional testing or model evaluation script
├── model/
│   ├── 1.h5                         # Trained Keras model
│   └── labels1.txt                  # Class labels for gestures
└── README.md                        # Documentation
```

---

## ⚙️ Installation

Make sure Python 3.x is installed. Then install dependencies:

```bash
pip install opencv-python cvzone numpy tensorflow
```

---

## 🧪 Scripts Overview

### `datacollection.py`

* Used for capturing hand gesture images from the webcam.
* Saves cropped gesture images for each class.

### `datanew.py`

* Likely used for training or processing the collected dataset.
* Can include normalization, model training, etc.

### `new.py`

* **Main script**: Runs live webcam gesture recognition.
* Uses `cvzone.HandTrackingModule` and `cvzone.ClassificationModule`.
* Displays Gujarati labels like:

  * "હેલો" (Hello)
  * "આભાર" (Thank You)

### `test.py`

* May be used to test model predictions on static images or saved data.

---

## 🚀 Run the Project

1. **Collect Data** (optional):

   ```bash
   python datacollection.py
   ```

2. **Train or Preprocess** (optional):

   ```bash
   python datanew.py
   ```

3. **Run Live Gesture Recognition**:

   ```bash
   python new.py
   ```

---

## 🧠 Model

* Model: `1.h5`
* Labels: `labels1.txt`
* Both are used by `cvzone.ClassificationModule` for prediction.

---

## 📌 Features

* Real-time gesture recognition
* Displays Gujarati text based on gestures
* Easy to extend with new gestures and retrain model

---

## 📈 Future Enhancements

* Add more gesture classes
* GUI interface with sound or text-to-speech
* Deploy as a web or mobile app

---

## 🙏 Acknowledgments

* [cvzone](https://github.com/cvzone/cvzone)
* TensorFlow & OpenCV communities

---

# Suspicious Human Activity Recognition using Deep Learning (UCF50)

This project focuses on recognizing human actions from video using a Long-term Recurrent Convolutional Network (LRCN) architecture. It is tailored for real-time suspicious activity detection with alerting and reporting features.

---

## 1. Dataset Used
- **Dataset**: [UCF50](https://www.kaggle.com/datasets/pypiahmad/realistic-action-recognition-ucf50) – a widely used benchmark dataset for human action recognition.
- **Selected Classes**: 
  - Walking  
  - Running  
  - Fighting  
  - TaiChi  

---

## 2. Data Preprocessing
- Extracted **20 evenly spaced frames** per video using **OpenCV**.
- Resized each frame to **64×64** pixels.
- Normalized pixel values to the **[0, 1]** range.
- Maintained **temporal consistency** for LSTM-based sequence modeling.

---

## 3. Models Implemented
- **ConvLSTM (Commented)**:  
  - Initially explored a **Convolutional LSTM** model to jointly learn spatial and temporal features.

- **LRCN (Final Model)**:
  - Used **TimeDistributed Conv2D** to extract frame-wise spatial features.
  - Followed by an **LSTM** to model temporal dynamics across frames.
  - Final **Dense + Softmax** layer to classify one of the 4 actions.

---

## 4. Training Strategy
- **Loss Function**: Categorical Crossentropy  
- **Optimizer**: Adam  
- **Early Stopping**: Enabled to prevent overfitting  
- **Train-Test Split**: 75% training, 25% testing

---

## 5. Evaluation
- Achieved 87% accuracy on the test set.
- Calculated **video-wise confidence** using model predictions on unseen clips.

---

## 6. Real-Time Prediction
- `predict_single_action(video_path)`  
  → Predicts action from a single video clip.

- `predict_on_video(video_path, output_path)`  
  → Annotates and saves a new video with predicted action overlay.

---

## 7. Alert System (Unique Feature)
- **Speech Alert**: Integrated using `pyttsx3` for suspicious activities.  
- **Email Alert**: Sends email using `smtplib` and credentials from `config.json`.  
- **PDF Report**: Automatically generated using `reportlab` with predicted actions.

---

## 8. Tools & Libraries Used
- **TensorFlow / Keras** – Deep learning model  
- **OpenCV** – Frame extraction and video handling  
- **Matplotlib** – Visualization  
- **MoviePy** – Video preview and writing  
- **ReportLab** – PDF report generation  
- **smtplib** – Email alerts  
- **pyttsx3** – Text-to-speech alerts

---

> ===> *This project is extensible to more action classes and can be integrated with live CCTV/RTSP feeds for real-world deployment.*

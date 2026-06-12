# ✍ Handwritten Character Recognition Using CNN

## 📌 Project Overview
This project is a Handwritten Character Recognition System built using PyTorch and Streamlit. The application allows users to upload an image containing a handwritten English alphabet character and predicts the corresponding letter (A–Z) using a Convolutional Neural Network (CNN).

## 🚀 Features
- Upload handwritten character images
- Predict English alphabets (A–Z)
- Deep Learning model using CNN
- Interactive Streamlit web interface
- Fast and accurate predictions

## 🛠️ Technologies Used
- Python
- PyTorch
- Streamlit
- Torchvision
- PIL (Pillow)

## 📂 Project Structure

```
Handwritten-Character-Recognition/
│
├── handwritten_recongtion.py
├── model.pth
├── README.md
└── requirements.txt
```

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/Handwritten-Character-Recognition.git
cd Handwritten-Character-Recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
streamlit run handwritten_recongtion.py
```

## 📸 How It Works
1. Upload an image containing a handwritten letter.
2. The image is preprocessed and resized to 28x28 pixels.
3. The CNN model analyzes the image.
4. The predicted alphabet is displayed on the screen.

## 🧠 Model Architecture
- Convolution Layer (32 Filters)
- ReLU Activation
- Max Pooling
- Convolution Layer (64 Filters)
- ReLU Activation
- Max Pooling
- Fully Connected Layer
- Output Layer (26 Classes)

## 🎯 Future Improvements
- Support handwritten words and sentences
- Add digit recognition (0–9)
- Improve accuracy with larger datasets
- Deploy on cloud platforms

## 👨‍💻 Author
Harshit Sharma

## ⭐ Support
If you found this project useful, consider giving it a star on GitHub.

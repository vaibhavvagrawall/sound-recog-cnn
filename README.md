# 🔊 Urban Sound Classification Web App

A Flask-based web application that classifies urban sounds (like sirens, dog barks, car horns, etc.) using a trained Convolutional Neural Network (CNN) model. Users can upload an audio file, and the app will predict the type of sound.

---

## 🚀 Demo

Upload an audio file (`.wav`, `.mp3`, etc.) and get an instant prediction of the sound category!

---

## 🎯 Features

- 🎵 Upload and classify audio files
- 🧠 Predict using a pre-trained CNN model
- 🔍 Detects 10 types of urban sounds
- 🌐 Built with Flask and TensorFlow
- 📁 Simple and clean UI using HTML

---

## 🔊 Sound Classes

The model can classify audio into the following 10 urban sound categories:

- Air Conditioner  
- Car Horn  
- Children Playing  
- Dog Bark  
- Drilling  
- Engine Idling  
- Gun Shot  
- Jackhammer  
- Siren  
- Street Music  

---

## 🧠 Model

- The model is a Convolutional Neural Network (CNN) trained on MFCC features extracted from audio files.
- The best-performing model is saved as `weights.best.basic_cnn.keras`.

---

## 🛠 Tech Stack

- **Frontend**: HTML5  
- **Backend**: Python, Flask  
- **Libraries**: TensorFlow, Librosa, NumPy  

---

## 📁 Folder Structure

```
urban-sound-classification/
├── pycache/                          # Python cache
├── app.py                            # Flask backend
├── utils.py                          # Helper functions for prediction
├── UrbanSound8K.csv                  # Dataset CSV file
├── Urban_Sound_Classification.ipynb  # Model training notebook
├── weights.best.basic_cnn.keras      # Trained CNN model
├── templates/
│   └── index.html                    # Frontend UI
├── uploads/
│   └── (uploaded audio files)
├── requirements.txt                  # Dependencies
└── README.md                         # Project documentation
```

---

## 📦 Installation

1. **Clone the repo**
```bash
git clone https://github.com/your-username/urban-sound-classification.git
cd urban-sound-classification
```

2. **Run the Flask app**
```bash
python app.py
```

3. **Visit the app in your browser**
```
http://127.0.0.1:5000/
```

---

## 🔍 Example

- Upload an audio clip (`.wav`, `.mp3`)
- Click **"Upload & Predict"**
- Get instant sound classification (e.g., `Dog Bark`, `Siren`, etc.)

---

## ✅ Requirements

- Flask  
- TensorFlow  
- Librosa  
- NumPy  

Install them via:

```bash
pip install Flask TensorFlow librosa numpy
```

---

## 📌 Future Improvements

- Add audio waveform or spectrogram visualization  
- Show model confidence scores  
- Deploy online (Render / Hugging Face / Vercel)  
- Add mobile responsiveness/support  

---

## 🤝 Contributing

Feel free to fork this repository and submit a pull request to improve the app!

---

## 👨‍💻 Author

**Anshuman Kumar**
[LinkedIn](https://linkedin.com/in/anshuman-kumar-372702282) • [GitHub](https://github.com/anshumankumar2021)

**Chirag Ranasaria**
[LinkedIn](https://linkedin.com/in/chirag-ranasaria-203281241) • [GitHub](https://github.com/Chirag9697)

**Vaibhav Agrawal**  
[LinkedIn](https://linkedin.com/in/vaibhavvagrawall) • [GitHub](https://github.com/vaibhavvagrawall)

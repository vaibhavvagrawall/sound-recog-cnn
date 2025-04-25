from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from utils import extract_features

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model
model = tf.keras.models.load_model('weights.best.basic_cnn.keras')

# Sound class labels (update these based on your dataset)
classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
           'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"✅ File saved at: {filepath}")
            print(f"📦 File size: {os.path.getsize(filepath)} bytes")

            try:
                features = extract_features(filepath)
                print(f"📐 Feature shape: {features.shape}")

                # Get prediction probabilities
                probs = model.predict(features)[0]
                print("🔍 Prediction probabilities:")
                for i, p in enumerate(probs):
                    print(f"  {classes[i]}: {p:.4f}")

                prediction_idx = np.argmax(probs)
                prediction = classes[prediction_idx]
                print(f"🎯 Final prediction: {prediction}")

            except Exception as e:
                print("❌ Error during prediction:", e)
                prediction = "Error processing the file. Please try again."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)

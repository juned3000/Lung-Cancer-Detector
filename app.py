from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model # type: ignore

app = Flask(__name__)

# Load the pre-trained model
model = load_model('lung_cancer_detector_model.keras')

def prepare_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        file_path = 'temp_image.png'
        file.save(file_path)

        img = prepare_image(file_path)
        prediction = model.predict(img)
        result = 'Cancerous' if prediction[0][0] > 0.5 else 'Non-Cancerous'
        confidence = float(prediction[0][0]) if result == 'Malignant' else float(1 - prediction[0][0])
        
        os.remove(file_path)

        return jsonify({'prediction': result, 'confidence': f"{confidence:.2%}"})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
import torch
from flask_cors import CORS
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import os

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Loading the model and feature extractor
model_name = "Sheetalavi/SHITAL_Leaf_Identification"
model = AutoModelForImageClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

def predict_image(image_path):
    # Loading and preprocessing the image
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Perform prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class label
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class_label = model.config.id2label[predicted_class_idx]

    return predicted_class_label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Get the prediction
            prediction = predict_image(file_path)
            return render_template('index.html', prediction=prediction, image=file.filename)

    return render_template('index.html')

# API endpoint to return predictions in JSON format
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Get the prediction
    prediction = predict_image(file_path)
    return jsonify({"prediction": prediction, "image_url": url_for('static', filename='uploads/' + file.filename)})

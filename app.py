import os
import torch
from flask import Flask, request, render_template, redirect, send_from_directory
from werkzeug.utils import secure_filename
from model import Net, CNN
from predict import preprocess_image, extract_digits, predict_digits, visualize_sequence
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the model
model = Net()
# model = CNN()
model.load_state_dict(torch.load('mnist_cnn1.pth'))
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            contours, image = preprocess_image(file_path)
            result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_contours.png')
            visualize_sequence(image, contours, save_path=result_image_path)
            digits = extract_digits(contours, image)
            predictions = predict_digits(digits, model)
            predictions_str = ''.join(map(str, predictions))
            return render_template('result.html', predictions=predictions_str, num_digits=len(digits), image_path=result_image_path)
    return render_template('upload.html')

@app.route('/draw', methods=['GET', 'POST'])
def draw():
    if request.method == 'POST':
        canvas_data = request.form['canvasData']
        image_data = base64.b64decode(canvas_data.split(',')[1])
        image = Image.open(BytesIO(image_data)).convert('L')
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'drawn_image.png')
        image.save(file_path)
        contours, image = preprocess_image(file_path)
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_contours.png')
        visualize_sequence(image, contours, save_path=result_image_path)
        digits = extract_digits(contours, image)
        predictions = predict_digits(digits, model)
        predictions_str = ''.join(map(str, predictions))
        return render_template('result.html', predictions=predictions_str, num_digits=len(digits), image_path=result_image_path)
    return render_template('draw.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

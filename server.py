from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model_path = 'models/basic_cnn.keras'
model = load_model(model_path)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Get the uploaded file from the request
    file = request.files['file']

    # Save the uploaded file to disk
    save_path = os.path.join('uploads', file.filename)
    file.save(save_path)

    # Load and preprocess the image
    img = image.load_img(save_path, target_size=(125, 125))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make a prediction using the loaded model
    prediction = model.predict(img_array)[0][0]

    # Determine the label based on the prediction
    if prediction >= 0.5:
        label = 'Malaria'
    else:
        label = 'Healthy'

    # Return the prediction as JSON
    return jsonify({'label': label, 'probability': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
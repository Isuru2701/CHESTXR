from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

CORS(app)
@app.route('/hello', methods=['post'])
def hello():
    return jsonify({'message': 'Hello World!'})


@app.route('/predict', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    #convert file to np 2d array
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    predictions = model.predict([preprocess_image(np.array(img))])
    predicted_labels = np.round(predictions).astype(int)
    predicted_labels

    print("PRED: ", predicted_labels)
    
    return jsonify({'prediction': 1 if predicted_labels[0][0] > 0.5 else 0})

def preprocess_image(image):
    # Read the image
    # Convert the image to RGB (if it's in BGR format)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE
    image = apply_clahe(image)
    # Resize the image to match the input size of the model
    image = cv2.resize(image, (224, 224))
    # Normalize the image
    image = image / 255.0
    # Expand the dimensions to create a batch of size 1
    image = np.expand_dims(image, axis=0)
    return image

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))



def apply_clahe(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
    image = clahe.apply(image)
    return image



def apply_sharpen(img:np.array, core=9):
    """

    applies the CLAHE enhancement over a given 2D array

    Params:
    img: image as a NumPy array

    Returns:
    normalized image
    """
    kernel = np.array([[-1,-1,-1], [-1,core,-1], [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

model = load_model('model/chest_xray_cnn_model.h5')

def predict(image):
    prediction = model.predict(image)
    return prediction.tolist()  # Convert to list for JSON serialization




if __name__ == '__main__':
    app.run(debug=True)

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
    file = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = preprocess_image(file)
    prediction = predict(image)
    print(prediction)
    
    return jsonify({'prediction': 1 if prediction[0][0] > 0.5 else 0})

@app.route('/yolo', methods=['POST'])
def yolo_model():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    #convert file to np 2d array
    file = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

def preprocess_image(image:np.array):
    # Resize to 256x256
    image = cv2.resize(image, (256, 256))
    cv2.imshow('image', image)
    #Apply CLAHE
    image = apply_CLAHE(image)
    #Apply Sharpen
    image = apply_sharpen(image)    
    # Normalize to [0, 1]
    image = image / 255.0
    # Reshape to (256, 256, 1)
    image = np.reshape(image, (256, 256, 1))
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    

    
    return image

def apply_CLAHE(img:np.array):
  """

  applies the CLAHE enhancement over a given 2D array

  Params:
    img: image as a NumPy array

  Returns:
    equalized image
  """

  src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  return clahe.apply(src)



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

model = load_model('model/CXR_best_model.h5')

def predict(image):
    prediction = model.predict(image)
    return prediction.tolist()  # Convert to list for JSON serialization




if __name__ == '__main__':
    app.run(debug=True)

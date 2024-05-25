from flask import Flask, request, jsonify
from model import predict
from preprocess import preprocess_image

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image = preprocess_image(file)
    prediction = predict(image)
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

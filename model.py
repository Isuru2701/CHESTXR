import numpy as np
from tensorflow.keras.models import load_model
import preprocess_image

# Load your pre-trained CNN model
model = load_model('model/CXR_best_model.h5')

def predict(image):
    prediction = model.predict(image)
    return prediction.tolist()  # Convert to list for JSON serialization

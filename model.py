import numpy as np
from tensorflow.keras.models import load_model

# Load your pre-trained CNN model
model = load_model('path_to_your_model.h5')

def predict(image):
    image = np.expand_dims(image, axis=0)  # Expand dims to fit model input shape
    prediction = model.predict(image)
    return prediction.tolist()  # Convert to list for JSON serialization

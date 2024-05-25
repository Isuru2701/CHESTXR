from PIL import Image
import numpy as np
import cv2

def preprocess_image(image:np.array):
    # Resize to 256x256
    image = cv2.resize(image, (256, 256))
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


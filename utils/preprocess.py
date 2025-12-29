import cv2
import numpy as np
def preprocess_ecg(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = img.reshape(1,224,224,1)
    return img
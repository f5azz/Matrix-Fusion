import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("model/plant_model.h5")

class_names = [
    "Tomato_Leaf_Curl",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Healthy",
    "Grape_Black_Rot",
    "Apple_Scab"
]

def predict(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img = image.img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    confidence = float(np.max(pred))
    label = class_names[np.argmax(pred)]

    return label, confidence
import os
from tensorflow import keras
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing import image
import numpy as np
import json

# Load the trained models
model_cnn = load_model('models/model_CNN.h5')
model_resnet = load_model('models/model_resnet.h5')
model_densenet = load_model('models/model_densenet.h5')

# Define the test image directory
test_dir = 'chest_xray/test/'
result_dir = 'static/test_results/'

# Ensure the result directory exists
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def predict_and_save_results(img_path, models, result_dir):
    results = {}
    print(f"Processing image: {img_path}")  # Debug: print the image path being processed
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    for model_name, model in models.items():
        prediction = model.predict(img_array)
        labels = ['NORMAL', 'PNEUMONIA']
        label = labels[1] if prediction > 0.5 else labels[0]
        probability = prediction[0][0] if prediction > 0.5 else 1 - prediction[0][0]
        results[model_name] = {"label": label, "probability": float(probability)}
        print(f"Model: {model_name}, Label: {label}, Probability: {probability}")  # Debug: print the prediction result

    # Save results to a JSON file
    img_name = os.path.basename(img_path)
    result_path = os.path.join(result_dir, img_name + '.json')
    with open(result_path, 'w') as f:
        json.dump(results, f)
    
    return results

models = {"CNN": model_cnn, "ResNet": model_resnet, "DenseNet": model_densenet}

# Iterate through test images and predict
for img_file in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_file)
    if os.path.isfile(img_path):  # Check if it's a file
        predict_and_save_results(img_path, models, result_dir)

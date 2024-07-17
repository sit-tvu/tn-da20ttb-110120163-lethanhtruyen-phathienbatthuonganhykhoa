from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow import keras
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained models
model_cnn = load_model('models/model_CNN.h5')
model_resnet = load_model('models/model_resnet.h5')
model_densenet = load_model('models/model_densenet.h5')

def predict_label(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)

    labels = ['NORMAL', 'PNEUMONIA']
    label = labels[1] if prediction > 0.5 else labels[0]
    

    return label

@app.route("/", methods=['GET', 'POST'])
def home():
    img_path = None
    results = {}
    file_name = None
    if request.method == 'POST':
        img = request.files['image']
        file_name = secure_filename(img.filename)
        img_path = "static/" + file_name
        img.save(img_path)

        for model_name, model in zip(["CNN", "ResNet", "DenseNet"], [model_cnn, model_resnet, model_densenet]):
            label = predict_label(img_path, model)
            results[model_name] = {"label": label}
        
    return render_template("index.html", results=results, img_path=img_path, file_name=file_name)

if __name__ == "__main__":
    app.run(debug=True)

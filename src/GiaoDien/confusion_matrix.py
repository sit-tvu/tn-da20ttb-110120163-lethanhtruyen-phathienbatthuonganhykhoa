import os
from tensorflow import keras
from keras._tf_keras.keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

model_CNN = load_model('models/model_CNN.h5')
model_densenet = load_model('models/model_densenet.h5')
model_resnet = load_model('models/model_resnet.h5')

dataset_path = 'chest_xray'
test_path = os.path.join(dataset_path, 'test')

batch_size = 64
image_size = (150, 150)
test_ds = keras.utils.image_dataset_from_directory(
    directory = test_path,
    image_size = image_size,
    batch_size = batch_size,
    label_mode = 'binary'
)

models = {
    "CNN": model_CNN,
    "ResNet": model_resnet,
    "DenseNet": model_densenet
}

def conf_matr(model, test_ds):
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.round(predictions).flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return cm

def plot_conf_matr(cm, model_name):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.title(f'Ma trận nhầm lẫn của mô hình {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.show()

for model_name, model in models.items():
    cm = conf_matr(model, test_ds)
    print(f"Ma trận nhầm lẫn của mô hình {model_name}:")
    print(cm)
    plot_conf_matr(cm, model_name)
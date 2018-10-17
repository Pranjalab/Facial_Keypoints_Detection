from keras.models import model_from_json
from keras import metrics
from Network import load
import numpy as np
import os


json = 'weights/1000.json'
h5 = 'weights/1000.h5'

# Model reconstruction from JSON file
with open(json, 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(h5)
model.compile(optimizer='adagrad', loss='mean_squared_error', metrics=['accuracy', metrics.mean_squared_error])

x_submit, _ = load(False)





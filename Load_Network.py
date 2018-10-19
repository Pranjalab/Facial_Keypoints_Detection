from keras.models import model_from_json
from keras import metrics
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import os


def load(train_flag=True):
    Train = 'data/training.csv'
    Test = 'data/test.csv'

    if train_flag:
        df = read_csv(os.path.expanduser(Train))
    else:
        df = read_csv(os.path.expanduser(Test))

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    print(df.count())
    df = df.dropna()

    print(df.count())

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if train_flag:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    X = X.reshape(-1, 96, 96, 1)
    return X, y

def display_pre(data, predict):
    y = predict
    image = data.reshape(96, 96)
    plt.imshow(image, cmap='gray')
    plt.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='o', s=10, c='r')


def load_model():
    json = 'weights/150.json'
    h5 = 'weights/150.h5'

    # Model reconstruction from JSON file
    with open(json, 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(h5)
    model.compile(optimizer='adagrad', loss='mean_squared_error', metrics=['accuracy', metrics.mean_squared_error])
    return model


if __name__ == '__main__':

    x_submit, _ = load(False)
    model = load_model()
    predicts = model.predict(x_submit)

    i = 0
    display_pre(x_submit[i], predicts[i])


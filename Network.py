# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 21:20:59 2018

@author: Pranjal
"""

import os
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import PyMail


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


from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection  import train_test_split
from keras.callbacks import TensorBoard
from keras import regularizers, metrics
from time import time
import pickle

x, y = load()
x_submit, _ = load(False)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1)

l2 = regularizers.l2(None)
l1 = regularizers.l1(None)
dropout = 0.2

batch_size = 10

np_epochs = 10

# Initialising the CNN
classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape=x_train[0].shape, activation='relu'))
classifier.add(Convolution2D(32, 3, 3, activation='relu',kernel_regularizer=l2, activity_regularizer=l1))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(64, 3, 3, activation='relu',kernel_regularizer=l2, activity_regularizer=l1))
classifier.add(Convolution2D(64, 3, 3, activation='relu',kernel_regularizer=l2, activity_regularizer=l1))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(128, 3, 3, activation='relu',kernel_regularizer=l2, activity_regularizer=l1))
classifier.add(Convolution2D(128, 3, 3, activation='relu',kernel_regularizer=l2, activity_regularizer=l1))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())
classifier.add(Dense(output_dim=512, activation='relu',kernel_regularizer=l2, activity_regularizer=l1))
classifier.add(Dropout(dropout))
classifier.add(Dense(output_dim=256, activation='relu',kernel_regularizer=l2, activity_regularizer=l1))
classifier.add(Dropout(dropout))
classifier.add(Dense(output_dim=128, activation='relu',kernel_regularizer=l2, activity_regularizer=l1))
classifier.add(Dropout(dropout))
classifier.add(Dense(output_dim=64, activation='relu',kernel_regularizer=l2, activity_regularizer=l1))
classifier.add(Dropout(dropout))
classifier.add(Dense(output_dim=30, activation='softmax'))

# Compiling the CNN
classifier.compile(optimizer='adagrad', loss='mean_squared_error', metrics=[metrics.mean_squared_error])
 
classifier.summary()

# Plot the graph
if not os.path.exists('TFlogs'):
    os.makedirs('TFlogs')


tensorboard = TensorBoard(log_dir='TFlogs/logs/{}'.format(time()))

history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, 
                         epochs=np_epochs, verbose=1, callbacks=[tensorboard], validation_data=(x_test, y_test))

metrics = classifier.evaluate(x_test, y_test, batch_size=batch_size)


# serialize model to JSON
if not os.path.exists('weights'):
    os.makedirs('weights')

model_json = classifier.to_json()
with open("weights/" + str(np_epochs) + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("weights/" + str(np_epochs)  + ".h5")
print("Saved model to disk")

# Store .pckl File
f = open('weights/' + str(np_epochs) + '.pckl', 'wb')
pickle.dump(history.history, f)
f.close()

# Get Notification Using PyMail
pymail = PyMail.pymail()
pymail.set_sent_address('pranjalab@gmail.com')
pymail.set_subject("Training Complete with {} epochs".format(str(np_epochs)))
pymail.set_body("Accuracy: " + str(metrics[0]) + "%\n\nLoss:\n\n" + str(metrics[1]))
pymail.send_mail()



































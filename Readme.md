# Facial_Keypoints_Detection
The project is the solution of [kaggle Facial Keypoints Detection problem](https://www.kaggle.com/c/facial-keypoints-detection) using CNN. get the dataset [here](https://drive.google.com/open?id=1jUxi6fHTO4H9Xw7PWa0mMcyutihhOQAg)
## Introduction
The objective of this task is to predict keypoint positions on face images. This can be used as a building block in several applications, such as:
- tracking faces in images and video
- analysing facial expressions
- detecting dysmorphic facial signs for medical diagnosis
- biometrics / face recognition

## Dependencies
- numpy
- pandas
- sklearn
- keras
- pickle
## Expiation
There are two main files in this project Network.py and Load_Network.py. Network.py file creates & saves the
networks weight files in .h5 format, Load_network.py file loads the saved weights for prediction.

### Network.py

In this file after importing all the dependencies a _"load"_ function is defend which readed _test.csv_ and _train.csv_
and return in pandas data frame, then the data is splited into training and test set.
An Convolutional neural network is created and trained with following architecture:

```terminal
_________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        conv2d_1 (Conv2D)            (None, 94, 94, 32)        320
        _________________________________________________________________
        conv2d_2 (Conv2D)            (None, 92, 92, 32)        9248
        _________________________________________________________________
        max_pooling2d_1 (MaxPooling2 (None, 46, 46, 32)        0
        _________________________________________________________________
        conv2d_3 (Conv2D)            (None, 45, 45, 64)        8256
        _________________________________________________________________
        max_pooling2d_2 (MaxPooling2 (None, 22, 22, 64)        0
        _________________________________________________________________
        conv2d_4 (Conv2D)            (None, 21, 21, 128)       32896
        _________________________________________________________________
        max_pooling2d_3 (MaxPooling2 (None, 10, 10, 128)       0
        _________________________________________________________________
        flatten_1 (Flatten)          (None, 12800)             0
        _________________________________________________________________
        dropout_1 (Dropout)          (None, 12800)             0
        _________________________________________________________________
        dense_1 (Dense)              (None, 500)               6400500
        _________________________________________________________________
        dropout_2 (Dropout)          (None, 500)               0
        _________________________________________________________________
        dense_2 (Dense)              (None, 500)               250500
        _________________________________________________________________
        dropout_3 (Dropout)          (None, 500)               0
        _________________________________________________________________
        dense_3 (Dense)              (None, 30)                15030
        =================================================================
        Total params: 6,716,750
        Trainable params: 6,716,750
        Non-trainable params: 0
```
while training callbacks of tensorboard is called to see accuracy and loss graph.
After training all the weights are saved.
#### (Optional)
following code in the end of the file can be used for sanding mail to you email address when training completes,
 to compile it for your project refer to my Pran_pymail project [here](https://github.com/Pranjalab/PyMail)
 ```python
import PyMail
pymail = PyMail.pymail()
pymail.set_sent_address('pranjalab@gmail.com')
pymail.set_subject("Training Complete with {} epochs".format(str(np_epochs)))
pymail.set_body("Accuracy: " + str(metrics[0]) + "%\n\nLoss:\n\n" + str(metrics[1]))
pymail.send_mail()
 ```

### Load_Network.py
In this file we load pre-trained weights and predict values. _display_pre_ function can be used to visualise predicted values.
![Alt text](Figure_1.png?raw=true "Title")

_save_pre_ is used to save the predicted value in _submission.csv_ file.

## Result
### Achieved score: 2.98340
## Acknowledgements
The data set for this competition was graciously provided by Dr. Yoshua Bengio of the University of Montreal to Kaggle.
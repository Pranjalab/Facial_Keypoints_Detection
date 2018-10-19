import cv2
import numpy as np
from Load_Network import load_model

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')

model = load_model()

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Video_1.mp4')
ret = True

while(ret):
    ret, frame = cap.read()
    frame = rescale_frame(frame, 100)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    if not faces == ():
        for (x, y, w, h) in faces:
            color = (255, 0, 0)  # BGR 0-255
            stroke = 2
            shape = (96/2, 96/2)
            end_cord_x = x + w
            end_cord_y = y + h
            center = (x + w/2, y + h/2)
            new_cod = (int(center[1] - shape[1]), int(center[1] + shape[1]),
                       int(center[0] - shape[0]), int(center[0] + shape[0]))
            face_image = gray[new_cod[0]:new_cod[1], new_cod[2]:new_cod[3]]
            (xf, yf, a) = (96,96,3)
            image = face_image.reshape(1,96,96,1)
            print(image.shape)
            predicts = model.predict(image)
            predict = predicts[0]
            x_arr = (predict[0::2] * xf/2)+ xf/2
            y_arr = (predict[1::2] * yf/2) + yf/2

            for i in range(len(x_arr)):
                cv2.circle(face_image, (x_arr[i],y_arr[i]), 2, (0,0,255), -1)
            # cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame', face_image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


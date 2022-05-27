import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import threading

# load model
model = load_model("best_model.h5")

listAnger = ['a1', 'a2']
listDisgust = ['d1', 'd2']
listFear = ['f1', 'f2', 'f3']
listHappy = ['h1', 'h2', 'h3']
listSad = ['s1', 's2']
listSurprise = ['S1', 'S2']
listNeutral = ['n1', 'n2']

predicted_emotion = ''


def giveSong():
    blank_image = np.ones([512, 512, 3], dtype=np.uint8)
    blank_image.fill(255)
    while True:
        global predicted_emotion
        # white blank image
        if predicted_emotion == 'angry':
            cv2.putText(blank_image, random.choice(listAnger), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif predicted_emotion == 'disgust':
            cv2.putText(blank_image, random.choice(listDisgust), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        2)
        elif predicted_emotion == 'fear':
            cv2.putText(blank_image, random.choice(listFear), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif predicted_emotion == 'happy':
            cv2.putText(blank_image, random.choice(listHappy), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif predicted_emotion == 'sad':
            cv2.putText(blank_image, random.choice(listSad), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif predicted_emotion == 'surprise':
            cv2.putText(blank_image, random.choice(listSurprise), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        2)
        else:
            cv2.putText(blank_image, random.choice(listNeutral), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        2)
        cv2.imshow("White Blank", blank_image)


face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

# thread = threading.Thread(target=giveSong)
# thread.start()

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows
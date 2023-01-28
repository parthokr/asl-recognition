import math

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np

imgSize = 500
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, )
classifier = Classifier("model/keras_model.h5", "model/labels.txt")
count = 1

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    # print(hands)
    cv2.imshow('image', img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        # crop image to boundary box
        imgCrop = img[y-20:y+h+20, x-20:x+w+20]
        cv2.imshow('img_cropped', imgCrop)

        # aspect ratio = h/w
        # if aspect ratio > 1
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        if h > w:
            w = math.ceil(imgSize * w/h)
            imgCrop = cv2.resize(imgCrop, (w, imgSize))
            pushWidth = math.ceil((imgSize-w)/2)
            imgWhite[:, pushWidth:pushWidth+w] = imgCrop
        else:
            h = math.ceil(imgSize * h/w)
            imgCrop = cv2.resize(imgCrop, (imgSize, h))
            pushHeight = math.ceil((imgSize-h)/2)
            imgWhite[pushHeight:pushHeight+h, :] = imgCrop
        # create a matrix of white image
        prediction, index = classifier.getPrediction(imgWhite)
        print(prediction, index)
        cv2.imshow('img_white', imgWhite)

    key = cv2.waitKey(1)
    if key == ord('s'):
        print(count)
        cv2.imwrite(f'data/C/{count}.jpg', imgWhite)
        count += 1
    elif key == ord('q'):
        break

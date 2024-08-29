import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset =20
imgsize = 30
counter = 0

folder = "C:/Users/Admin/Desktop/Sign language detection/Data/Thank you"
while True :
    success, img = cap.read ()
    hands , img = detector.findHands (img)
    if hands :
        hand = hands [0]
        x,y,w,h =hand['bbox']

        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255

        imgcrop = img[y-offset : y+h+ offset , x-offset: x + w + offset]
        imgcropshape = imgcrop.shape

        aspectratio = h/w

        if aspectratio >1:
            k =imgsize/h
            wCal =math.ceil(k*w)
            imgresize = cv2.resize(imgcrop, (wCal,imgsize))
            imgresizeshape =imgresize.shape
            wGap = math.ceil((imgsize -wCal)/2)
            imgwhite[:, wGap: wCal + wGap] = imgresize

        else:
            k = imgsize / w
            hCal = math.ceil(k * h)
            imgresize = cv2.resize(imgcrop, (imgsize, hCal))
            imgresizeShape = imgresize.shape
            hGap = math.ceil((imgsize - hCal) / 2)
            imgwhite[hGap: hCal + hGap, :] = imgresize

        cv2.imshow('ImageCrop', imgcrop)
        cv2.imshow('ImageWhite', imgwhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgwhite)
        print(counter)


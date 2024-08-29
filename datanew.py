import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize the camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgsize = 300
counter = 0

# Folder path where images will be saved
folder = "C:/Users/Admin/Desktop/Sign language detection/Data/A"

# Function to capture an image after a countdown
def capture_image_after_countdown(img, imgwhite):
    for i in range(5, 0, -1):
        imgCountdown = img.copy()
        cv2.putText(imgCountdown, f"Capturing in {i} seconds", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Image', imgCountdown)
        cv2.waitKey(1000)  # Wait for 1 second

    return imgwhite

# Main loop to capture 30 images
while counter < 30:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]  # Access the first detected hand
        x, y, w, h = hand['bbox']

        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255

        imgcrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgcropshape = imgcrop.shape

        aspectratio = h / w

        if aspectratio > 1:
            k = imgsize / h
            wCal = math.ceil(k * w)
            imgresize = cv2.resize(imgcrop, (wCal, imgsize))
            imgresizeshape = imgresize.shape
            wGap = math.ceil((imgsize - wCal) / 2)
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

        # Automatically capture an image after 5 seconds
        imgwhite = capture_image_after_countdown(img, imgwhite)
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgwhite)
        print(f"Image {counter} saved.")

    cv2.imshow('Image', img)
    
    # Exit if the user presses the 'q' key
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

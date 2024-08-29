import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from PIL import ImageFont, ImageDraw, Image

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("C:\\Users\\Admin\\Desktop\\converted keras\\1.h5", "C:\\Users\\Admin\\Desktop\\converted keras\\labels1.txt")
offset = 20
imgSize = 300
counter = 0

# Define Gujarati labels
labels = ["હેલો", "આભાર"]

# Load Gujarati font (Ensure this path is correct and font is installed)
font_path = "C:\\Users\\Admin\\Desktop\\Sign language detection\\Rasa-VariableFont_wght.ttf"  # Replace with the path to your Gujarati font file
font = ImageFont.truetype(font_path, 40)  # Adjust font size as needed

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Convert OpenCV image to PIL image
        pil_img = Image.fromarray(cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Draw text with PIL (Gujarati)
        draw.rectangle((x - offset, y - offset - 70, x - offset + 400, y - offset + 60 - 50), fill=(0, 255, 0))
        draw.text((x, y - 30), labels[index], font=font, fill=(0, 0, 0))

        # Convert back to OpenCV format
        imgOutput = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

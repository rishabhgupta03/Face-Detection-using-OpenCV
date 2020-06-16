import cv2 as cv
import numpy as np
original_image = cv.imread("4.jpg)
gray_scale = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")
detected_faces = face_cascade.detectMultiScale(gray_scale, 1.3)

for (column, row, width, height) in detected_faces:
    cv.rectangle(
        original_image,
        (column, row),
        (column + width, row+height),
        (0, 0, 255),
        3
    )

small = cv.resize(original_image, (0,0), fx=0.58, fy=0.58)
cv.imshow('image', small)
cv.waitKey(0)
cv.destroyAllWindows()

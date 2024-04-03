import cv2
import imutils
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('digit_recognition_model_MNIST.h5')

# Get the reference to the webcam
CAMERA = cv2.VideoCapture(0)

CAPTURE_WIDTH = 900
ROI_LONG = 400  # Region Of Interest
MARGIN = 200
TOP = MARGIN
RIGHT = CAPTURE_WIDTH - MARGIN
BOTTOM = TOP + ROI_LONG
LEFT = RIGHT - ROI_LONG

while True:
    _, frame = CAMERA.read()
    frame = imutils.resize(frame, width=CAPTURE_WIDTH)
    (height, width) = frame.shape[:2]

    # Add rectangle to original frame
    cv2.rectangle(frame, (LEFT, TOP), (RIGHT, BOTTOM), (0, 255, 0), 2)

    # Cut ROI and preprocess
    roi = frame[TOP+2:BOTTOM-2, LEFT+2:RIGHT-2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)  # Need fixes
    cv2.imshow("ROI", gray)

    # Resize and preprocess the image for prediction
    gray_small = imutils.resize(gray, width=28)
    gray_small = gray_small.reshape(1, 28, 28, 1)
    gray_small = gray_small / 255.0  # Normalize pixel values

    # Predict and show prediction
    pred_probs = model.predict(gray_small)[0]
    pred = np.argmax(pred_probs)
    LABEL_TEXT = str(pred)
    LABEL_COLOR = (0, 255, 0)
    cv2.putText(frame, LABEL_TEXT, (LEFT, TOP-7), cv2.FONT_HERSHEY_SIMPLEX, 1, LABEL_COLOR, 2)
    cv2.imshow("Frame", frame)

    # If the user pressed "q", then stop looping
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all windows
CAMERA.release()
cv2.destroyAllWindows()
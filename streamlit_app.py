import cv2
import time
import numpy as np
import threading as th
import concurrent.futures as cf
import streamlit as st
from keras.models import load_model

cnn_model = load_model('cnn_model.h5')
folders = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Create a video capture object
cap = cv2.VideoCapture(0)

# Set the video resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

st.title("Virtual Sign Language Interpreter")

# Create a canvas to display the video frames
canvas = st.empty()
text_area = st.empty()
stop = st.button(label="Stop", key="end_live_stream")

# Function to detect action in a video frame
def detect_action(last_letter, letter):
    ret, frame = cap.read()

    img_for_pred = cv2.resize(frame, dsize=(64, 64)).astype('float32')
    result_arr = cnn_model.predict(np.expand_dims(img_for_pred, axis=0))[0]
    letter = folders[np.argmax(result_arr)]

    last_letter+=letter

    return last_letter, letter

# To stream the live camera
def live_stream():
    ret, frame = cap.read()

    if not ret:
        st.error("Error reading the video stream.")
        return

    # Check if the user has clicked the 'Stop' button
    if stop:
        print("clicked stop")
        return frame, False
    
    return frame, True

letter = ""
last_letter = ""
while True:

    with cf.ThreadPoolExecutor() as executor:
        detect_action_result = executor.submit(detect_action, last_letter, letter)
        last_letter, letter = detect_action_result.result()
        text_area.write(last_letter)

        live_stream_result = executor.submit(live_stream)
        frame, result = live_stream_result.result()
        canvas.image(frame, channels="BGR")

        time.sleep(2)
        if not result:
            detect_action_result.cancel()
            live_stream_result.cancel()
            break

# Release the video capture object and close the Streamlit app
cap.release()
cv2.destroyAllWindows()
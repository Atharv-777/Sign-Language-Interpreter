from flask import Flask, render_template, Response
from keras.models import load_model
import numpy as np
import cv2
import os

app = Flask('__name__', template_folder="templates")

last_letter = ""
model = load_model(r"C:/project/cnn_model.h5")

path = r"C:\project\asl_alphabet_train\asl_alphabet_train"

folders = os.listdir(path)
extra_letter = ['del', 'space', 'nothing']
extra_letter_path = []
for letter in folders:
    if letter in extra_letter:
        extra_letter_path.append(letter)
        folders.remove(letter)
folders.sort()

folders += extra_letter_path
img_classes = [os.path.join(path, letter) for letter in folders]
# print(img_classes)

@app.route("/")
def index():
    return render_template("index.html")

class Var:
    def __init__(self):
        self.current_letter = ""

    def setVar(self, letter):
        self.current_letter = letter
        # return self.current_letter

var = Var()

@app.route("/predict_letter")
def predict_letter():
    # video = cv2.VideoCapture(0)
    # _, frame = video.read()
    # output = frame.copy()
    # frame_for_pred = cv2.resize(frame, (64, 64)).astype('float32')
    # result_arr = model.predict(np.expand_dims(frame_for_pred, axis=0))[0]
    # predicted_letter = folders[np.argmax(result_arr)]
    # video.release()
    # cv2.destroyAllWindows()
    # return predict_letter
    print("button clicked...")

def gen_frames():
    video = cv2.VideoCapture(0)
    global current_letter
    current_letter = ""
    while True:
        success, frame = video.read()
        if success:
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            except Exception as e:
                pass
        
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/stt")
def stt():
    return render_template('stt.html')

@app.route("/tts")
def tts():
    return render_template("tts.html")

if __name__=="__main__":
    app.run

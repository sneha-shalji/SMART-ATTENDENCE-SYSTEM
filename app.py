from flask import Flask, render_template, Response
import numpy as np
import cv2
import face_recognition
import csv
from datetime import datetime

import os
print("Current working directory:", os.getcwd())


app = Flask(__name__)



anya_img = face_recognition.load_image_file('static\Anya.jpg')
anya_encoding = face_recognition.face_encodings(anya_img)[0]

florence_img = face_recognition.load_image_file('static\Florence.jpg')
florence_encoding = face_recognition.face_encodings(florence_img)[0]

rdj_img = face_recognition.load_image_file('static\Rdj.jpg')
rdj_encoding = face_recognition.face_encodings(rdj_img)[0]

tom_image = face_recognition.load_image_file('static\Tom.jpg')
tom_encoding = face_recognition.face_encodings(tom_image)[0]

srk_image = face_recognition.load_image_file('static\srk.jpg')
srk_encoding = face_recognition.face_encodings(srk_image)[0]

brad_image = face_recognition.load_image_file('static\Brad.jpg')
brad_encoding = face_recognition.face_encodings(brad_image)[0]
 
kanye_image = face_recognition.load_image_file('static\kanye.jpg')
kanye_encoding = face_recognition.face_encodings(kanye_image)[0]


known_face_encoding = [
    anya_encoding, 
    florence_encoding,
    rdj_encoding,
    tom_encoding,
    srk_encoding,
    brad_encoding,
    kanye_encoding
]

known_face_names = [
    "Anya Taylor Joy",
    "Florence Pugh",
    "Robert Downey jr",
    "Tom Cruise",
    "Shah Rukh Khan",
    "Brad Pitt",
    "Kanye West"
]


students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+',newline = '')
lnwrite = csv.writer(f)

cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Resize frame
            small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distance)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                    if name in students:
                        students.remove(name)
                        current_time = datetime.now().strftime("%H:%M:%S")
                        lnwrite.writerow([name, current_time])

                    # Draw on frame
                    cv2.putText(frame, name, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

            # Encode frame to display on web
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
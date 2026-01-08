import numpy as np
import cv2
import face_recognition
import csv
from datetime import datetime

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

        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            name = "Unknown"

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                if name in students:
                    students.remove(name)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    lnwrite.writerow([name, current_time])
                    f.flush()

            # Scale back face locations
            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        yield frame

while True:
    frame = next(generate_frames())
    cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
f.close()
cv2.destroyAllWindows()
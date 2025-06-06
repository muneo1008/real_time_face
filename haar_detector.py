import cv2
from PIL.ImageOps import scale


def apply_effect(frame, x, y, w, h, mode="box"):
    roi = frame[y:y+h, x:x+w]
    if mode == "mosaic":
        roi = cv2.resize(roi, (10, 10), interpolation=cv2.INTER_LINEAR)
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[y:y+h, x:x+w] = roi
    elif mode == "blur":
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        frame[y:y+h, x:x+w] = roi
    else:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame

def detect_faces(frame, mode="box",min_confidence=0.5):

    #감지 얼굴 수
    face_cnt = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    scale = 1.1 + ( 1.0 - min_confidence) * 0.5
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_cnt += 1
        frame = apply_effect(frame, x, y, w, h, mode)
    return frame, face_cnt

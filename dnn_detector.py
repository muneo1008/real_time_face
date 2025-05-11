import cv2
import numpy as np

# 모델
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

#효과 적용
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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

#얼굴 탐지
def detect_faces(frame, mode="box", min_confidence=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), False, False)
    net.setInput(blob)
    detections = net.forward()

    #감지 얼굴 수
    face_cnt = 0


    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            face_cnt += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            print('신뢰도: ',confidence,', 좌표: ', x1, y1, x2, y2)
            frame = apply_effect(frame, x1, y1, x2 - x1, y2 - y1, mode)
    return frame, face_cnt

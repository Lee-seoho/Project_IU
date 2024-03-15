import cv2
from facenet_pytorch import MTCNN
from keras.models import load_model
import numpy as np
import torch
import face_recognition

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# mtcnn = MTCNN(keep_all=True, device=device)

# IU_face.h5 모델 로드
classification_model = load_model('C:/Users/white/Desktop/iu_project/IU_face_pretrained.h5')

cap = cv2.VideoCapture('test2.mp4')

frame_skip = 5
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # boxes, scores = mtcnn.detect(frame)
    boxes = face_recognition.face_locations(frame)

    if boxes is not None:
        for box in boxes:
            top, right, bottom, left = box[0], box[1], box[2], box[3]

            # 얼굴 이미지 추출 및 전처리
            face_image = frame[top:bottom, left:right]

            if not face_image.size:
                # 얼굴 이미지가 비어 있다면 처리하지 않음
                continue
            face_image = cv2.resize(face_image, (70, 70))
            face_image = face_image.astype('float32') / 255.0
            face_image = np.expand_dims(face_image, axis=0)

            # 라벨 예측
            result = classification_model.predict(face_image)
            print("result", result)

            prediction = np.argmax(classification_model.predict(face_image), axis=1)[0]
            label = "IU detected" if prediction == 0 else "Other person detected"
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)

            # 각 얼굴에 라벨 표시
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 얼굴 영역 표시
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

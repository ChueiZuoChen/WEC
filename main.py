import cv2
import numpy as np
if __name__ == '__main__':
    detector = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
    recog = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    ids = []

    for i in range(1, 29):
        img = cv2.imread(f'face01/{i}.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_np = np.array(gray, 'uint8')
        face = detector.detectMultiScale(gray)
        for (x, y, w, h) in face:
            faces.append(img_np[y:y + h, x:x + w])
            ids.append(1)

    for i in range(1, 31):
        img = cv2.imread(f'face02/{i}.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_np = np.array(gray, 'uint8')
        face = detector.detectMultiScale(gray)
        for (x, y, w, h) in face:
            faces.append(img_np[y:y + h, x:x + w])
            ids.append(2)

    print('training model...')
    recog.train(faces, np.array(ids))
    recog.save('face.yml')
    print('ok!')


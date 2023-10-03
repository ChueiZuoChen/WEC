from os import name

import cv2
import mediapipe as mp

if __name__ == '__main__':
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('face.yml')
    cascade_path = "xml/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
    ) as face_detection:
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            ret, img = cap.read()
            if not ret:
                print("Cannot receive frame")
                break
            size = img.shape
            w = size[1]
            h = size[0]
            img = cv2.resize(img, (540, 300))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray)
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detection.process(img2)

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(img, detection)
                    s = detection.location_data.relative_bounding_box
                    eye = int(s.width * w * 0.1)
                    a = detection.location_data.relative_keypoints[0]
                    b = detection.location_data.relative_keypoints[1]
                    ax, ay = int(a.x * w), int(a.y * h)
                    bx, by = int(b.x * w), int(b.y * h)
                    cv2.circle(img, (ax, ay), (eye + 10), (255, 255, 255), -1)
                    cv2.circle(img, (bx, by), (eye + 10), (255, 255, 255), -1)
                    cv2.circle(img, (ax, ay), eye, (0, 0, 0), -1)
                    cv2.circle(img, (bx, by), eye, (0, 0, 0), -1)
                    cv2.circle(img, (ax - 8, ay - 8), (eye - 15), (255, 255, 255), -1)
                    cv2.circle(img, (bx - 8, by - 8), (eye - 15), (255, 255, 255), -1)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                if confidence < 50:
                    text = name[str(idnum)]
                else:
                    text = '???'
                cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Potti', img)
            if cv2.waitKey(5) == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

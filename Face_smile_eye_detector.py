import cv2

face_detector_cv = cv2.CascadeClassifier('Files/haarcascade_frontalface_default.xml')
smile_detector_cv = cv2.CascadeClassifier('Files/haarcascade_smile.xml')
eye_detector_cv = cv2.CascadeClassifier('Files/haarcascade_eye.xml')

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = face_detector_cv.detectMultiScale(grayscale_frame, scaleFactor=1.7, minNeighbors=5)
    # smile_coordinates = smile_detector_cv.detectMultiScale(grayscale_frame, scaleFactor=1.7, minNeighbors=20)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (104, 223, 57), 1)

        detected_face = frame[y:y+h, x:x+w]
        grayscale_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) 

        smile_coordinates = smile_detector_cv.detectMultiScale(grayscale_face, scaleFactor=1.7, minNeighbors=20)
        eye_coordinates = eye_detector_cv.detectMultiScale(grayscale_face)


        for (x_d, y_d, w_d, h_d) in smile_coordinates:
            cv2.rectangle(detected_face, (x_d, y_d), (x_d+w_d, y_d+h_d), (43, 148, 239), 1)

        if len(smile_coordinates) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+30), fontScale=1, fontFace=cv2.FONT_ITALIC, color=(0, 0, 0))

        for (x_d, y_d, w_d, h_d) in eye_coordinates:
            cv2.rectangle(detected_face, (x_d, y_d), (x_d+w_d, y_d+h_d), (255, 255, 255), 1)

        if len(eye_coordinates) > 1:
            cv2.putText(frame, 'Eyes detected', (x, y+h+60), fontScale=1, fontFace=cv2.FONT_ITALIC, color=(0, 0, 0))

    # for (x, y, w, h) in smile_coordinates:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (43, 148, 239), 1)

    cv2.imshow('Smile Detector app', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()
cv2.destroyAllWindows()
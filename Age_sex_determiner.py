import cv2
import math

age_model = "ML_projects/models/age_net.caffemodel"
age_proto = "ML_projects/models/age_deploy.prototxt"

gender_model = "ML_projects/models/gender_net.caffemodel"
gender_proto = "ML_projects/models/gender_deploy.prototxt"

face_model = "ML_projects/models/opencv_face_detector_uint8.pb"
face_proto = "ML_projects/models/opencv_face_detector.pbtxt"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_list = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)"
]

gender_list = ["Male", "Female"]

face_net = cv2.dnn.readNet(face_model, face_proto)
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

video = cv2.VideoCapture(0)

padding = 20


def detect_face(image):

    h, w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image,
        1.0,
        (300, 300),
        [104, 117, 123],
        True,
        False
    )

    face_net.setInput(blob)

    detections = face_net.forward()

    boxes = []

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:

            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            boxes.append([x1, y1, x2, y2])

    return boxes


while True:

    grabbed, frame = video.read()

    if not grabbed:
        break

    boxes = detect_face(frame)

    for box in boxes:

        x1, y1, x2, y2 = box

        face = frame[
            max(0, y1 - padding):
            min(y2 + padding, frame.shape[0] - 1),

            max(0, x1 - padding):
            min(x2 + padding, frame.shape[1] - 1)
        ]

        blob = cv2.dnn.blobFromImage(
            face,
            1.0,
            (227, 227),
            MODEL_MEAN_VALUES,
            swapRB=False
        )

        gender_net.setInput(blob)
        gender_preds = gender_net.forward()

        gender = gender_list[gender_preds[0].argmax()]

        age_net.setInput(blob)
        age_preds = age_net.forward()

        age = age_list[age_preds[0].argmax()]

        label = f"{gender}, {age}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.rectangle(
            frame,
            (x1, y1 - 35),
            (x2, y1),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

    cv2.imshow("Age and Gender Detection", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

video.release()
cv2.destroyAllWindows()
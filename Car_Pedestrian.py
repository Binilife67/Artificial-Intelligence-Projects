import cv2

car_detector_cv = cv2.CascadeClassifier('Files/cars_detector.xml')
pedestrian_detector_cv = cv2.CascadeClassifier('Files/haarcascade_fullbody.xml')

test_img = cv2.imread('Files/imtest5.jpeg')
test_img = cv2.imread('Files/imgtest1.png')
test_img = cv2.imread('Files/crosswalk-featured.jpg')
test_vid = cv2.VideoCapture('Files/F1_car_on_highway.mp4') # 0
test_vid = cv2.VideoCapture('Files/car_moving_on_aroad.mp4')

grayscale_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

car_coordinates = car_detector_cv.detectMultiScale(grayscale_img)
pedestrian_coordinates = car_detector_cv.detectMultiScale(grayscale_img)
# print(len(car_coordinates))

for (x, y, w, h) in car_coordinates:
    cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 255), 1)

for (x, y, w, h) in pedestrian_coordinates:
    cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 255, 0), 1)

cv2.imshow('Car and pedestrain Detector app', test_img)
cv2.waitKey()

# print(car_detector_cv.empty())
# print(test_vid.isOpened())

while True:
    successful_frame_read, frame = test_vid.read()
    if not successful_frame_read:
        break

    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    car_coordinates = car_detector_cv.detectMultiScale(grayscale_frame)
    pedestrian_coordinates = pedestrian_detector_cv.detectMultiScale(grayscale_frame)
    # car_coordinates = car_detector_cv.detectMultiScale(
    #     grayscale_frame,
    #     scaleFactor=1.1,
    #     minNeighbors=3,
    #     minSize=(60,60)
    # )

    for (x, y, w, h) in car_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 1)

    for (x, y, w, h) in pedestrian_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 1)

    cv2.imshow('Car and pedestrian detector app', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

test_vid.release()
cv2.destroyAllWindows()
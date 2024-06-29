import cv2
import sys

tracker = cv2.TrackerKCF_create()
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Could not open video")
    sys.exit()

ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()

#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def get_initial_bbox(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    face = faces[0]
    x_center = face[0] + face[2] // 2
    y_center = face[1] + face[3] // 2
    bbox = (x_center - 50, y_center - 50, 100, 100)
    return bbox

bbox = get_initial_bbox(frame)
if bbox is None:
    print('No face detected')
    sys.exit()

ok = tracker.init(frame, bbox)

while True:
    ok, frame = video.read()
    if not ok:
        break

    timer = cv2.getTickCount()

    ok, bbox = tracker.update(frame)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        bbox = get_initial_bbox(frame)
        if bbox is not None:
            tracker = cv2.TrackerKCF_create()  # Recreate the tracker
            ok = tracker.init(frame, bbox)

    cv2.putText(frame, "KCF Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    cv2.imshow("Tracking", frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
from ultralytics import YOLO 
import cv2
import cvzone
import math

feed = "rtsp://admin:Admin@123@192.168.16.28/Streaming/Channels/101"
cap = cv2.VideoCapture(0)

model = YOLO(r"C:\Users\deepasruthi.ramesh\Documents\Image_detection\yolo_with_webcam\yolo-Weights\yolov8n.pt")

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

while True:
    success, img = cap.read()
    if not success:
        print(" Failed to read RTSP stream. Check connection.")
        break

    results = model(img, stream=True)

    person_count = 0 

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100)) / 100

            if classNames[cls] == "person" and conf > 0.5:
                person_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'Person {conf}', (max(0, x1), max(35, y1)))

    cv2.putText(img, f"Person Count: {person_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("RTSP Feed - YOLOv8 Person Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)

model = YOLO("yolov8m1000.pt")
#8m300 best
#8s500 mid

classNames = ['Glasses', 'Hardhat', 'No Glasses', 'No Hardhat', 'No Safety West', 'NoWork Shoes', 'Safety West', 'Work Shoes']
    #["Hardhat", "No Glasses", "Safety West", "Glasses", "No Hardhat", "No Safety West", "Work Shoes",
             # "No Work Shoes"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)
            if conf > 0.4:
                if currentClass == 'No Glasses' or currentClass == 'No Hardhat' or\
                        currentClass == 'No Safety West' or currentClass == 'NoWork Shoes':
                            myColor = (0, 0, 255)
                elif currentClass == 'Hardhat' or currentClass == 'Safety West' or \
                        currentClass == 'Glasses' or currentClass == 'Work Shoes':
                            myColor = (255, 0, 0)
                else:
                    myColor = (255, 0, 0)
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                    (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                    colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # downloads automatically on first run
cap = cv2.VideoCapture(0)   # 0 = default webcam

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True, classes=[39], verbose = False)

    for r in results:
        if len(r.boxes) > 0:
            print("bottle detected")
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                if (x1 + x2) / 2 < width * 0.3:
                    print("RIGHT")
                elif (x1 + x2) / 2 > width * 0.7:
                    print("LEFT")
                else:
                    print("FORWARD")
            
        annotated_frame = r.plot()
        cv2.imshow("YOLO Bottle Detection", annotated_frame)

        
        
    if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
        break

cap.release()
cv2.destroyAllWindows()
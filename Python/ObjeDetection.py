from ultralytics import YOLO
import cv2
import torch


model = YOLO("EGGY_THINGS2/Dataset/CandlingV2.pt")  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

cap = cv2.VideoCapture(0)  # Change to 1 if using an external camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            conf = box.conf[0].item()  
            cls = int(box.cls[0]) 
            label = f"{model.names[cls]} {conf:.2f}"

 
            cv2.rectangle(frame, (x1, y1),  (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





from ultralytics import YOLO
import cv2
import torch


model = YOLO("./CandlingV2.pt")  # Change path if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

cap = cv2.VideoCapture(0)  # Change to 1 if using an external camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame, conf=0.5)

    # Draw bounding boxes on the frame
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0])  # Class index
            label = f"{model.names[cls]} {conf:.2f}"

            # Draw box and label
            cv2.rectangle(frame, (x1, y1),  (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show output
    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



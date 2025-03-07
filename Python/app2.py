
import cv2
from ultralytics import YOLO

model = YOLO('./candling.pt')
print(model.names)
webcamera1 = cv2.VideoCapture(0)


while True:
    success1, frame1 = webcamera1.read()
    
    results1 = model.track(frame1, conf=0.35, imgsz=480)
    
    print("NUM RESULTS")
    print(len(results1))
    
    cv2.putText(frame1, f"Total: {len(results1[0].boxes)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Live Camera", results1[0].plot())

    if cv2.waitKey(1) == ord('q'):
        break

webcamera1.release()
cv2.destroyAllWindows()
 


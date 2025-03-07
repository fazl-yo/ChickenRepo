import cv2
import pandas as pd
from ultralytics import YOLO
import datetime
today = datetime.datetime.now()


MODEL = YOLO(r"best.pt")

results = MODEL(r"./EggCandlingTest.jpg")
result_image = results[0].plot()

df = pd.read_csv("EggData.csv")

cell_size = 60
num_rows = result_image.shape[0] // cell_size
num_cols = result_image.shape[1] // cell_size


for i in range(num_rows + 1):
    cv2.line(result_image, (0, i * cell_size), (result_image.shape[1], i * cell_size), (0, 255, 0), 2)
for j in range(num_cols + 1):
    cv2.line(result_image, (j * cell_size, 0), (j * cell_size, result_image.shape[0]), (0, 255, 0), 2)

for i in range(num_rows):
    for j in range(num_cols):
        label = f"({i}, {j})"
        cv2.putText(result_image, label, (j * cell_size + 10, i * cell_size + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow('Result with Table', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
Fertile = "Fertile"
Infertile = "Infertile"
coordinates = []

for result in results:
    for detection in result.boxes:
        class_label = detection.cls
        if class_label := Fertile:
            x1, y1, x2, y2 = detection.xyxy[0]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            row = center_y // cell_size
            col = center_x // cell_size
            coordinates.append({
                'Row': row,
                'Column': col,
                'Center_X': center_x,
                'Center_Y': center_y,
                'Date': today.strftime("%Y-%m-%d"),
                'Time': today.strftime("%H:%M:%S"),
                'Status': Fertile
            })
        elif class_label := Infertile:
            x1, y1, x2, y2 = detection.xyxy[0]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            row = center_y // cell_size
            col = center_x // cell_size
            coordinates.append({
                'Row': row,
                'Column': col,
                'Center_X': center_x,
                'Center_Y': center_y,
                'Date': today.strftime("%Y-%m-%d"),
                'Time': today.strftime("%H:%M:%S"),
                'Status': Infertile
            })

for coord in coordinates:
    print(f"Fertile Egg at Row: {coord['Row']}, Column: {coord['Column']}, Center: ({coord['Center_X']}, {coord['Center_Y']})")

new_data = pd.DataFrame(coordinates)

new_data.to_csv("EggData.csv", mode='a', index=False, header=False)
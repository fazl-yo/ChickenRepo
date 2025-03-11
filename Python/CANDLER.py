import cv2
import torch
from ultralytics import YOLO
import pandas as pd
import datetime

model = YOLO(r"../Dataset/CandlingV2.pt")
cap = cv2.VideoCapture(0)

df = pd.read_csv("../Other/RESULTS.csv")

cell_size = 100  
class_positions = {}  
new_data = []  # List to store new data

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            label = model.names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Object: {label}, Coordinates: ({x1}, {y1}), Width: {x2 - x1}, Height: {y2 - y1}")

            col = x1 // cell_size
            row = y1 // cell_size
            if label not in class_positions:
                class_positions[label] = {'rows': set(), 'cols': set()}
            class_positions[label]['cols'].add(col)
            class_positions[label]['rows'].add(row)

            # Append new data to the list
            new_data.append({'Rows': row, 'Columns': col, 'Class': label})

    height, width, _ = frame.shape
    num_rows = height // cell_size
    num_cols = width // cell_size

    for i in range(num_rows + 1):
        cv2.line(frame, (0, i * cell_size), (width, i * cell_size), (0, 255, 0), 2)
    for j in range(num_cols + 1):
        cv2.line(frame, (j * cell_size, 0), (j * cell_size, height), (0, 255, 0), 2)

    for i in range(num_rows):
        for j in range(num_cols):
            label = f"({i}, {j})"
            cv2.putText(frame, label, (j * cell_size + 10, i * cell_size + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Object Detection with Grid', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        for label, positions in class_positions.items():
            positions_list = [(row, col) for row in positions['rows'] for col in positions['cols']]
            positions_str = ', '.join([f'("{row}", "{col}")' for row, col in positions_list])
            print(f"Class '{label}' appeared at {positions_str}")
        break

cap.release()
cv2.destroyAllWindows()

# Append new data to the DataFrame and save it back to the CSV file
new_df = pd.DataFrame(new_data)
df = pd.concat([df, new_df], ignore_index=True)
df.to_csv("../Other/RESULTS.csv", index=False)
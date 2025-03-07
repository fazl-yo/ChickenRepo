import cv2
import pandas as pd
from ultralytics import YOLO
import datetime
today = datetime.datetime.now()


MODEL = YOLO(r"candling.pt")

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


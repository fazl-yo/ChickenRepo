import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

cap = cv2.VideoCapture(0)

def read_numbers_from_image(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([55, 255, 68])  
    upper_green = np.array([85, 236, 68]) 


    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    

    inverted_mask = cv2.bitwise_not(mask)


    result = cv2.bitwise_and(image, image, mask=inverted_mask)


    gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ocr_result = pytesseract.image_to_string(thresh_image, config='--psm 6')

    numbers = ''.join([char for char in ocr_result if char.isdigit()])

    return numbers, thresh_image

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    detected_numbers, processed_image = read_numbers_from_image(frame)


    cv2.putText(frame, f"Detected Numbers: {detected_numbers}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    cv2.imshow('Original Camera Feed', frame)
    cv2.imshow('Processed Image', processed_image)  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


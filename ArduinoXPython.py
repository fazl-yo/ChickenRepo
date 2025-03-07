import serial
import time

arduino = serial.Serial('COM4', 9600, timeout=1)
time.sleep(2) 

def control_led(state):
    arduino.write(b'1' if state else b'0')

try:
    while True:
        control_led(True)  
        time.sleep(1)      
        control_led(False) 
        time.sleep(1)      
except KeyboardInterrupt:
    arduino.close()

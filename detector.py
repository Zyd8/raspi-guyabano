
import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
import tkinter as tk
import sys
import os
from tensorflow.keras.models import load_model
from threading import Thread, Lock

# ------------------ GPIO SETUP ------------------ #
TRIG = 5
ECHO = 6
servo_pin = 18
motor_in1 = 23
motor_in2 = 24
motor_en = 25
relay_pin = 21  # Relay pin added

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.setup(servo_pin, GPIO.OUT)
pwm = GPIO.PWM(servo_pin, 50)  # 50Hz
pwm.start(0)

GPIO.setup(motor_in1, GPIO.OUT)
GPIO.setup(motor_in2, GPIO.OUT)
GPIO.setup(motor_en, GPIO.OUT)
motor_pwm = GPIO.PWM(motor_en, 100)
motor_pwm.start(0)

GPIO.setup(relay_pin, GPIO.OUT)  # Relay setup
GPIO.output(relay_pin, GPIO.LOW)  # Initial state: OFF (adjust if needed)

# ------------------ GLOBAL FLAGS ------------------ #
gui_open = False
servo_started = False
gui_lock = Lock()

# ------------------ DISTANCE MEASUREMENT ------------------ #
def get_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 34300 / 2
    return round(distance, 2)

# ------------------ MOTOR CONTROL ------------------ #
def motor_forward(speed=50):
    
    GPIO.output(motor_in1, GPIO.HIGH)
    GPIO.output(motor_in2, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(speed)
    GPIO.output(relay_pin, GPIO.LOW)
    print("Motor moving forward.")

def motor_stop():
    GPIO.output(motor_in1, GPIO.LOW)
    GPIO.output(motor_in2, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(0)
    GPIO.output(relay_pin, GPIO.HIGH)
    print("Relay OFF")
    print("Motor stopped.")

# ------------------ SERVO CONTROL ------------------ #
def set_angle(angle):
    duty = angle / 18 + 2
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(0)

def rotate_servo_step_by_step():
    angles = [0, 45, 90, 135, 180]
    for angle in angles:
        set_angle(angle)
        print(f"Stopped at {angle} degrees")
        time.sleep(1)
    
    # Wait 1 second after last rotation
    print("Servo finished rotating. Waiting 1 second before turning ON the rela>
    time.sleep(1)

    # Turn ON the relay (change to LOW if your relay is active LOW)
    GPIO.output(relay_pin, GPIO.HIGH)
    print("Relay ON")

# ------------------ LOAD MODEL ------------------ #
model = load_model('model.h5')

# ------------------ GUI RESULT DISPLAY ------------------ #
def show_result_gui(label, accuracy):
    global gui_open
    with gui_lock:
        if gui_open:
            return
        gui_open = True

    def on_close():
       global gui_open
        with gui_lock:
            gui_open = False
        result_root.destroy()

    result_root = tk.Tk()
    result_root.title("Classification Result")
    tk.Label(result_root, text=f"Result: {label}", font=("Arial", 16)).pack(pad>
    tk.Label(result_root, text=f"Confidence: {accuracy * 100:.2f}%", font=("Ari>
    tk.Button(result_root, text="OK", command=on_close, font=("Arial", 12)).pac>
    result_root.protocol("WM_DELETE_WINDOW", on_close)
    result_root.mainloop()

# ------------------ DETECTION FUNCTION ------------------ #
def detect_ripe_tomatoes(frame, model):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SI>

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi = frame[y:y+h, x:x+w]
            roi = cv2.resize(roi, (224, 224))
            roi = roi / 255.0
            roi = np.expand_dims(roi, axis=0)
            prediction = model.predict(roi)[0][0]
            accuracy = prediction if prediction > 0.5 else 1 - prediction
            label = "Rotten" if prediction > 0.5 else "Fresh"

            if accuracy < 0.95:
                Thread(target=show_result_gui, args=(label, accuracy)).start()
            else:
                print(f"Detected '{label}' with high confidence ({accuracy * 10>

            break

    cv2.imshow("Detection", frame)

# ------------------ CAMERA SETUP ------------------ #
cap = cv2.VideoCapture(0)
data=False

# ------------------ DETECTION PROCESS ------------------ #
def start_camera_detection():
    global servo_started
    try:
        while True:
            dist = get_distance()
            print(f"Distance: {dist} cm")
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read frame")
                break

            if dist <= 50:
                motor_stop()
                print("Object too close! Motor stopped.")
                if not servo_started:
                    print("Starting servo rotation...")
                    Thread(target=rotate_servo_step_by_step).start()
                   servo_started = True
                    time.sleep(6)


                    print("Servo rotation completed. Now performing detection.")

            else:

                motor_forward(speed=60)
                detect_ripe_tomatoes(frame, model)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

# ------------------ GUI BUTTON FUNCTIONS ------------------ #
def on_start():
    start_button.config(state=tk.DISABLED)
    Thread(target=start_camera_detection, daemon=True).start()

def on_reset():
    print("Resetting system...")
    cap.release()
    cv2.destroyAllWindows()
    pwm.stop()
    motor_pwm.stop()
    GPIO.cleanup()
    python = sys.executable
 os.execl(python, python, *sys.argv)

def on_gui_close():
    print("Exiting GUI and cleaning up resources...")
    cap.release()
    cv2.destroyAllWindows()
    pwm.stop()
    motor_pwm.stop()
    GPIO.cleanup()
    main_root.destroy()

# ------------------ MAIN GUI ------------------ #
main_root = tk.Tk()
main_root.title("Guyabano Detection System")
main_root.geometry("300x250")

tk.Label(main_root, text="Guyabano Detection", font=("Arial", 16)).pack(pady=20)

start_button = tk.Button(main_root, text="Start Process", font=("Arial", 14), c>
start_button.pack(pady=10)

reset_button = tk.Button(main_root, text="Reset", font=("Arial", 14), command=o>
reset_button.pack(pady=10)

main_root.protocol("WM_DELETE_WINDOW", on_gui_close)
main_root.mainloop()

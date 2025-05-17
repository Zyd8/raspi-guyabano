import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
import tkinter as tk
import sys
import os

# Try importing TensorFlow, provide helpful error message if not available
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not found. Make sure it's installed correctly.")
    print("On Raspberry Pi, you might need to run: ")
    print("pip3 install tensorflow")
    print("or: sudo apt-get install python3-tensorflow")
    TENSORFLOW_AVAILABLE = False

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

# ------------------ SERVO CONTROL ------------------ #
def set_angle(angle):
    duty = angle / 18 + 2
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(0)


# Initialize servo and set to 0 degrees
GPIO.setup(servo_pin, GPIO.OUT)
pwm = GPIO.PWM(servo_pin, 50)  # 50Hz (20ms PWM period)
pwm.start(0)  # Start with 0% duty cycle
set_angle(0)  # Reset to 0 degrees on startup
print("Servo reset to 0 degrees on startup")

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
GPIO.output(relay_pin, GPIO.HIGH)  # Initial state: OFF (adjust if needed)

# ------------------ GLOBAL FLAGS ------------------ #
gui_open = False
servo_started = False
scanning_complete = False
gui_lock = Lock()
last_detection_time = None  # Will be initialized when start button is clicked
distance_label = None  # Will hold reference to the distance label
motor_running = False  # Track motor state
program_running = True  # Track if program is running

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
    global motor_running
    if not motor_running:
        GPIO.output(motor_in1, GPIO.HIGH)
        GPIO.output(motor_in2, GPIO.LOW)
        motor_pwm.ChangeDutyCycle(speed)
        GPIO.output(relay_pin, GPIO.LOW)
        motor_running = True
        print("Motor moving forward.")

def show_motor_stop_gui():
    global gui_open
    with gui_lock:
        if gui_open:
            return
        gui_open = True

    def on_close():
        global gui_open
        with gui_lock:
            gui_open = False
        stop_root.destroy()

    stop_root = tk.Tk()
    stop_root.title("Conveyor Status")
    tk.Label(stop_root, text="Conveyor has stopped", font=("Arial", 16)).pack(pady=10)
    tk.Button(stop_root, text="OK", command=on_close, font=("Arial", 12)).pack(pady=10)
    stop_root.protocol("WM_DELETE_WINDOW", on_close)
    stop_root.mainloop()

def motor_stop():
    global motor_running
    if motor_running:
        GPIO.output(motor_in1, GPIO.LOW)
        GPIO.output(motor_in2, GPIO.LOW)
        motor_pwm.ChangeDutyCycle(0)
        GPIO.output(relay_pin, GPIO.HIGH)
        motor_running = False
        print("Relay OFF")
        print("Motor stopped.")
        Thread(target=show_motor_stop_gui).start()

def rotate_servo_step_by_step():
    global servo_started, scanning_complete

    # Scan positions
    angles = [0, 45, 90, 135]
    predictions = []
    confidences = []
    
    for angle in angles:
        set_angle(angle)
        print(f"Scanning at {angle} degrees")
        time.sleep(1)  # Give time for the servo to settle

        # Capture image at each step and process it
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Couldn't capture image at {angle} degrees.")
            continue
            
        # Process the frame and get prediction
        if TENSORFLOW_AVAILABLE and model is not None:
            try:
                # Get the ROI (center 50% of the frame)
                height, width = frame.shape[:2]
                roi_size = 0.5
                x1 = int(width * (1 - roi_size) / 2)
                y1 = int(height * (1 - roi_size) / 2)
                x2 = int(width * (1 + roi_size) / 2)
                y2 = int(height * (1 + roi_size) / 2)
                roi = frame[y1:y2, x1:x2]
                
                # Prepare ROI for prediction
                roi = cv2.resize(roi, (224, 224))
                roi = roi / 255.0
                roi = np.expand_dims(roi, axis=0)
                
                # Make prediction
                prediction = model.predict(roi)[0][0]
                is_good_quality = prediction < 0.5
                confidence = 1 - prediction if is_good_quality else prediction
                
                # Store predictions and confidences
                predictions.append(is_good_quality)
                confidences.append(confidence)
                
                print(f"Angle {angle}°: {'Good' if is_good_quality else 'Bad'} quality ({confidence*100:.1f}%)")
                
            except Exception as e:
                print(f"Error during prediction at {angle}°: {e}")
    
    # After collecting all predictions, determine final quality
    if predictions:
        good_count = sum(predictions)
        bad_count = len(predictions) - good_count
        
        # Calculate average confidence for the majority class
        if good_count > bad_count:
            final_quality = "Good"
            avg_confidence = np.mean([conf for i, conf in enumerate(confidences) if predictions[i]])
        elif bad_count > good_count:
            final_quality = "Bad"
            avg_confidence = np.mean([conf for i, conf in enumerate(confidences) if not predictions[i]])
        else:  # tie
            final_quality = "Good"  # Default to good in case of tie
            avg_confidence = np.mean(confidences)
        
        print(f"\nFinal Quality Assessment:")
        print(f"- Good quality views: {good_count}")
        print(f"- Bad quality views: {bad_count}")
        print(f"- Final Decision: {final_quality} quality ({avg_confidence*100:.1f}% confidence)")
        
        # Show the final result in the GUI
        Thread(target=show_result_gui, args=(f"{final_quality} Quality", avg_confidence)).start()
    else:
        print("No valid predictions were made during the scan.")
    
    # Set scanning complete flag
    scanning_complete = True

    # Wait 1 second after last rotation
    print("Servo finished rotating. Waiting 1 second before turning ON the relay.")
    time.sleep(1)

    # Turn ON the relay
    GPIO.output(relay_pin, GPIO.HIGH)
    print("Relay ON")

    # Return to starting position (0 degrees)
    print("Returning to starting position...")
    set_angle(0)

    # Turn relay back off
    GPIO.output(relay_pin, GPIO.LOW)
    print("Relay OFF")

    # Indicate that scanning is complete and reset servo flag
    scanning_complete = True
    servo_started = False
    print("Servo returned to starting position. Ready for next scan.")
    
    # Add delay to prevent rescanning the same object
    print("Waiting 3 seconds before allowing next scan...")
    time.sleep(3)
    print("Ready for next object.")

# ------------------ LOAD MODEL ------------------ #
if TENSORFLOW_AVAILABLE:
    try:
        model = load_model('model.h5')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        TENSORFLOW_AVAILABLE = False
else:
    model = None
    print("Running without TensorFlow model - detection functionality will be limited")

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
    tk.Label(result_root, text=f"Result: {label}", font=("Arial", 16)).pack(pady=10)
    tk.Label(result_root, text=f"Confidence: {accuracy * 100:.2f}%", font=("Arial", 14)).pack(pady=10)
    tk.Button(result_root, text="OK", command=on_close, font=("Arial", 12)).pack(pady=10)
    result_root.protocol("WM_DELETE_WINDOW", on_close)
    result_root.mainloop()

# ------------------ DETECTION FUNCTION ------------------ #
def detect_guyabano(frame, model):
    global last_detection_time
    
    # Make a copy of the frame to draw on
    processed_frame = frame.copy()
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Define a region of interest (center 50% of the frame)
    roi_size = 0.5
    x1 = int(width * (1 - roi_size) / 2)
    y1 = int(height * (1 - roi_size) / 2)
    x2 = int(width * (1 + roi_size) / 2)
    y2 = int(height * (1 + roi_size) / 2)
    
    # Draw ROI rectangle for visualization
    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Only attempt prediction if TensorFlow is available
    if TENSORFLOW_AVAILABLE and model is not None:
        try:
            # Get the ROI and process it for the model
            roi = frame[y1:y2, x1:x2]
            roi = cv2.resize(roi, (224, 224))
            roi = roi / 255.0
            roi = np.expand_dims(roi, axis=0)
            
            # Make prediction
            prediction = model.predict(roi)[0][0]
            accuracy = prediction if prediction > 0.5 else 1 - prediction
            label = "Bad Quality" if prediction > 0.5 else "Good Quality"
            
            # Add label and confidence to the frame
            label_text = f"{label} ({accuracy*100:.1f}%)"
            cv2.putText(processed_frame, label_text, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show GUI result if confidence is not high
            if accuracy < 0.95:
                Thread(target=show_result_gui, args=(label, accuracy)).start()
            else:
                print(f"Detected '{label}' with high confidence ({accuracy * 100:.2f}%)")
                
            # Object detected, reset the timer
            last_detection_time = time.time()
                
        except Exception as e:
            print(f"Error during prediction: {e}")
            cv2.putText(processed_frame, "Detection Error", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return processed_frame

# ------------------ CAMERA SETUP ------------------ #
cap = cv2.VideoCapture(0)
data=False

# ------------------ CLEANUP FUNCTION ------------------ #
def cleanup():
    global program_running
    program_running = False
    print("Cleaning up resources...")
    # Stop the motor
    motor_stop()
    # Stop the servo
    pwm.ChangeDutyCycle(0)
    # Release camera
    cap.release()
    # Clean up GPIO
    GPIO.cleanup()
    print("Cleanup complete.")

# ------------------ DETECTION PROCESS ------------------ #
def start_camera_detection():
    global servo_started, scanning_complete, last_detection_time, program_running
    try:
        while program_running:
            dist = get_distance()
            print(f"Distance: {dist} cm")
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read frame")
                break
                
            # Check for timeout (45 seconds without detection)
            if last_detection_time is not None:  # Only check timeout if detection has started
                current_time = time.time()
                if current_time - last_detection_time > 45:
                    print("No object detected for 45 seconds. Stopping program.")
                    motor_stop()
                    print("Conveyor stopped due to timeout.")
                    time.sleep(1)
                    return

            if dist <= 36 or dist >= 140:
                motor_stop()
                print("Object detected! Checking if it's a guyabano...")
                last_detection_time = time.time()
                rotate_servo_step_by_step()
                # First check if it's a guyabano
                is_guyabano = False
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_red = np.array([0, 100, 100])
                upper_red = np.array([10, 255, 255])
                mask = cv2.inRange(hsv, lower_red, upper_red)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:
                        is_guyabano = True
                        break
                
                if is_guyabano:
                    print("Guyabano detected! Starting scanning process...")
                    if not servo_started:
                        # Process the frame with detect_guyabano and get the result
                        frame = detect_guyabano(frame, model)
                        servo_started = True
                        scanning_complete = False
                        Thread(target=rotate_servo_step_by_step).start()
                    
                    if scanning_complete:
                        scanning_complete = False
                        print("Scan complete. Ready to continue.")
                        motor_forward(speed=70)
                        time.sleep(2)
                        
                else:
                    print("No guyabano detected. Continuing conveyor...")
                    motor_forward(speed=70)
                    time.sleep(2)
            else:
                motor_forward(speed=70)
                
            # Display the video feed
            cv2.imshow('Camera Feed', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cleanup()  # Ensure cleanup happens even if there's an error

# Add signal handler for graceful shutdown
import signal
def signal_handler(signum, frame):
    print("\nReceived signal to terminate. Cleaning up...")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ------------------ GUI BUTTON FUNCTIONS ------------------ #
def on_start():
    global last_detection_time, program_running
    program_running = True
    last_detection_time = time.time()  # Initialize timeout counter when start button is clicked
    start_button.config(state=tk.DISABLED)
    Thread(target=start_camera_detection, daemon=True).start()

def on_reset():
    print("Resetting system...")
    global servo_started, scanning_complete, last_detection_time, motor_running
    servo_started = False
    scanning_complete = False
    last_detection_time = None  # Reset the timeout counter
    motor_running = False  # Reset motor state
    # Return servo to starting position
    set_angle(0)
    GPIO.output(relay_pin, GPIO.LOW)
    start_button.config(state=tk.NORMAL)
    print("System reset complete.")

def on_gui_close():
    print("Exiting GUI and cleaning up resources...")
    cleanup()
    main_root.destroy()

def update_distance_label():
    global distance_label
    while True:
        if distance_label is not None:
            dist = get_distance()
            distance_label.config(text=f"Distance: {dist:.1f} cm")
        time.sleep(0.1)  # Update every 100ms

# ------------------ MAIN GUI ------------------ #
main_root = tk.Tk()
main_root.title("Guyabano Detection System")
main_root.geometry("300x300")

tk.Label(main_root, text="Guyabano Detection", font=("Arial", 16)).pack(pady=20)

distance_label = tk.Label(main_root, text="Distance: -- cm", font=("Arial", 14))
distance_label.pack(pady=10)

start_button = tk.Button(main_root, text="Start Process", font=("Arial", 14), command=on_start)
start_button.pack(pady=10)

reset_button = tk.Button(main_root, text="Reset", font=("Arial", 14), command=on_reset)
reset_button.pack(pady=10)

# Start the distance update thread
distance_thread = Thread(target=update_distance_label, daemon=True)
distance_thread.start()

main_root.protocol("WM_DELETE_WINDOW", on_gui_close)
main_root.mainloop()


import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
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
relay_pin = 21

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(motor_in1, GPIO.OUT)
GPIO.setup(motor_in2, GPIO.OUT)
GPIO.setup(motor_en, GPIO.OUT)
GPIO.setup(relay_pin, GPIO.OUT)
GPIO.output(relay_pin, GPIO.HIGH)
# Initialize PWM for motor speed control
motor_pwm = GPIO.PWM(motor_en, 100)
motor_pwm.start(0)

# Initialize servo
GPIO.setup(servo_pin, GPIO.OUT)
pwm = GPIO.PWM(servo_pin, 50)  # 50Hz for servo
pwm.start(0)

def set_angle(angle, delay=0.5):
    """Set servo angle with optional delay and turn off the signal to prevent jitter"""
    duty = angle / 18 + 2
    pwm.ChangeDutyCycle(duty)
    time.sleep(delay)
    pwm.ChangeDutyCycle(0)  # Turn off the signal to prevent jitter
    
# Servo movement parameters
current_angle = 0  # Track current angle (0-180)
scan_direction = 1  # 1 for increasing angle, -1 for decreasing
SCAN_ANGLES = [0, 45, 90, 135, 180]  # Define scan positions

# Initialize servo to 0 position
set_angle(0, 1.0)  # Give extra time for initial positioning

# ------------------ GLOBAL FLAGS ------------------ #
program_running = False
servo_started = False
scanning_complete = False
gui_open = False
last_detection_time = None
motor_running = False
cap = None
model = None
gui_lock = Lock()
last_scan_time = 0  # Timestamp of last scan completion
SCAN_COOLDOWN = 5  # Cooldown period in seconds
result_text = ""  # Store the latest quality result

# GUI elements
root = None
video_label = None
distance_label = None
status_label = None
start_button = None
stop_button = None
reset_button = None

# ------------------ MOTOR CONTROL ------------------ #
def motor_forward(speed=50):
    global motor_running
    if not motor_running:
        GPIO.output(motor_in1, GPIO.HIGH)
        GPIO.output(motor_in2, GPIO.LOW)
        motor_pwm.ChangeDutyCycle(speed)
        GPIO.output(relay_pin, GPIO.LOW)
        motor_running = True

def motor_stop():
    global motor_running
    if motor_running:
        GPIO.output(motor_in1, GPIO.LOW)
        GPIO.output(motor_in2, GPIO.LOW)
        motor_pwm.ChangeDutyCycle(0)
        GPIO.output(relay_pin, GPIO.HIGH)
        motor_running = False

# ------------------ SERVO CONTROL ------------------ #
def set_angle(angle, delay=0.5):
    """Set servo angle with optional delay and turn off the signal to prevent jitter"""
    duty = angle / 18 + 2
    pwm.ChangeDutyCycle(duty)
    time.sleep(delay)
    pwm.ChangeDutyCycle(0)  # Turn off the signal to prevent jitter

def rotate_servo_step_by_step():
    global servo_started, scanning_complete, program_running, cap, model, motor_running, current_angle, scan_direction
    
    # Initialize scan_direction if not set
    if 'scan_direction' not in globals():
        global scan_direction
        scan_direction = 1  # Default to forward direction
    
    if not program_running:
        return
        
    # Ensure motor is stopped before starting scan
    motor_stop()
    
    # Determine the next set of angles based on current direction
    if scan_direction == 1:
        angles = SCAN_ANGLES  # Forward scan: 0° to 180°
    else:
        angles = SCAN_ANGLES[::-1]  # Reverse scan: 180° to 0°
    
    # Skip the first angle if it's the same as current (except for 0°)
    if angles[0] == current_angle and current_angle != 0:
        angles = angles[1:]
    
    predictions = []
    confidences = []
    
    try:
        for angle in angles:
            if not program_running:
                break
                
            # Update status
            if status_label:
                status_label.config(text=f"Status: Scanning at {angle}°...")
                
            # Calculate movement parameters
            angle_diff = abs(angle - current_angle)
            move_delay = min(0.5 + (angle_diff * 0.005), 1.0)  # Slightly longer delay for larger moves
            
            # Move servo to position with dynamic delay
            set_angle(angle, move_delay)
            current_angle = angle  # Update current angle
            
            # Skip processing for the first and last positions when just moving
            if (angle == 0 or angle == 180) and len(predictions) > 0:
                time.sleep(0.3)  # Shorter delay for end positions
                continue
                
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Process frame if model is available
            if TENSORFLOW_AVAILABLE and model is not None:
                try:
                    # Get center ROI of the frame
                    height, width = frame.shape[:2]
                    roi_size = 0.5
                    x1 = int(width * (1 - roi_size) / 2)
                    y1 = int(height * (1 - roi_size) / 2)
                    x2 = int(width * (1 + roi_size) / 2)
                    y2 = int(height * (1 + roi_size) / 2)
                    roi = frame[y1:y2, x1:x2]
                    
                    # Prepare image for model
                    roi = cv2.resize(roi, (224, 224))
                    roi = roi / 255.0
                    roi = np.expand_dims(roi, axis=0)
                    
                    # Get prediction
                    prediction = model.predict(roi)[0][0]
                    is_good_quality = prediction < 0.5
                    confidence = 1 - prediction if is_good_quality else prediction
                    
                    # Store results
                    predictions.append(is_good_quality)
                    confidences.append(confidence)
                    
                    print(f"Angle {angle}°: {'Good' if is_good_quality else 'Bad'} quality ({confidence*100:.1f}%)")
                    
                except Exception as e:
                    print(f"Error during prediction at {angle}°: {e}")
        
        # Process results if we have any predictions
        if predictions:
            good_count = sum(predictions)
            bad_count = len(predictions) - good_count
            
            if good_count > bad_count:
                final_quality = "Good"
                avg_confidence = np.mean([conf for i, conf in enumerate(confidences) if predictions[i]])
            elif bad_count > good_count:
                final_quality = "Bad"
                avg_confidence = np.mean([conf for i, conf in enumerate(confidences) if not predictions[i]])
            else:
                final_quality = "Good"  # Default to good in case of tie
                avg_confidence = np.mean(confidences)
            
            global result_text
            result_text = f"{final_quality} Quality ({avg_confidence*100:.1f}%)"
            print(f"Final quality assessment: {result_text}")
            
            # Actuate relay based on quality (example: activate for bad quality)
            if final_quality == "Bad":
                GPIO.output(relay_pin, GPIO.HIGH)
                print("Actuating relay for bad quality")
                time.sleep(1)  # Keep relay on for 1 second
                GPIO.output(relay_pin, GPIO.LOW)
            
# Toggle scan direction for next time
            scan_direction *= -1
        else:
            if status_label:
                status_label.config(text="Status: No valid predictions")
    
    except Exception as e:
        print(f"Error during servo scan: {e}")
        if status_label:
            status_label.config(text=f"Status: Error during scan - {str(e)}")
    
    finally:
        # Always ensure we clean up properly
        try:
            # Ensure we're at 0 or 180 degrees based on direction (in case of early exit)
            if scan_direction == 1:
                set_angle(0, 1.0)  # Extra time for end positions
                current_angle = 0
            else:
                set_angle(180, 1.0)
                current_angle = 180
            
            # Ensure relay is off
            GPIO.output(relay_pin, GPIO.LOW)
            
            # Update scanning state
            scanning_complete = True
            servo_started = False
            
            # Small delay before allowing next scan
            time.sleep(5)
            
            # Reset scanning state
            servo_started = False
            scanning_complete = False
            
            # Update status with quality result and set last scan time
            last_scan_time = time.time()
            
            # Keep the quality result visible for longer
            QUALITY_DISPLAY_TIME = 5  # Show quality result for 5 seconds
            READY_DELAY = 2  # Additional delay before showing "Ready"
            
            if status_label:
                # First show the quality result
                status_label.config(text=f"Status: {result_text} (cooldown for {SCAN_COOLDOWN}s)")
                
                # After QUALITY_DISPLAY_TIME, show "Ready" status
                if 'root' in globals() and root is not None:
                    def show_ready_status():
                        if status_label:
                            status_label.config(text=f"Status: Ready (cooldown for {SCAN_COOLDOWN}s)")
                    
                    # After SCAN_COOLDOWN + READY_DELAY, show "Ready" and start motor if needed
                    def final_ready():
                        if status_label:
                            status_label.config(text="Status: Ready")
                        dist = get_distance()
                        if dist is not None and (dist > 36 and dist < 140):
                            motor_forward(speed=70)
                    
                    # Schedule the status updates
                    root.after(QUALITY_DISPLAY_TIME * 1000, show_ready_status)
                    root.after((SCAN_COOLDOWN + READY_DELAY) * 1000, final_ready)
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
            last_scan_time = time.time()  # Still update last_scan_time on error to prevent rapid retries

# ------------------ DISTANCE MEASUREMENT ------------------ #
def get_distance():
    pulse_start = time.time()
    pulse_end = time.time()
    timeout = 0.1
    try:
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
        start_time = time.time()
        while GPIO.input(ECHO) == 0 and (time.time() - start_time) < timeout:
            pulse_start = time.time()
        start_time = time.time()
        while GPIO.input(ECHO) == 1 and (time.time() - start_time) < timeout:
            pulse_end = time.time()
        pulse_duration = pulse_end - pulse_start
        distance = (pulse_duration * 34300) / 2
        return round(distance, 2) if distance > 0 else None
    except Exception as e:
        print(f"Error in distance measurement: {e}")
        return None

# ------------------ DETECTION FUNCTION ------------------ #
def detect_guyabano(frame, model):
    global last_detection_time
    processed_frame = frame.copy()
    height, width = frame.shape[:2]
    roi_size = 0.5
    x1 = int(width * (1 - roi_size) / 2)
    y1 = int(height * (1 - roi_size) / 2)
    x2 = int(width * (1 + roi_size) / 2)
    y2 = int(height * (1 + roi_size) / 2)
    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if TENSORFLOW_AVAILABLE and model is not None:
        try:
            roi = frame[y1:y2, x1:x2]
            roi = cv2.resize(roi, (224, 224))
            roi = roi / 255.0
            roi = np.expand_dims(roi, axis=0)
            prediction = model.predict(roi)[0][0]
            accuracy = prediction if prediction > 0.5 else 1 - prediction
            label = "Bad Quality" if prediction > 0.5 else "Good Quality"
            label_text = f"{label} ({accuracy*100:.1f}%)"
            cv2.putText(processed_frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            last_detection_time = time.time()
        except Exception as e:
            print(f"Error during prediction: {e}")
            cv2.putText(processed_frame, "Detection Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return processed_frame

# ------------------ VIDEO FEED UPDATE FOR TKINTER ------------------ #
def update_video_feed():
    global program_running, video_label, cap, distance_label, status_label, servo_started, scanning_complete, last_detection_time, motor_running, last_scan_time
    if program_running and cap is not None and cap.isOpened():
        dist = get_distance()
        if distance_label:
            distance_label.config(text=f"Distance: {dist if dist is not None else '--'} cm")
        
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            display_frame = frame.copy()
            
            # Check if we should process new object detection
            current_time = time.time()
            cooldown_elapsed = (current_time - last_scan_time) > SCAN_COOLDOWN
            
            # Only process if not in the middle of a scan and cooldown has elapsed
            if not servo_started and not scanning_complete and cooldown_elapsed:
                # If no object detected and motor isn't running, start the motor
                if dist is not None and (dist > 36 and dist < 140) and not motor_running:
                    motor_forward(speed=70)
                # If object detected, stop and process it
                elif dist is not None and (dist <= 36 or dist >= 140) and not servo_started:
                    motor_stop()
                    status_label.config(text="Status: Object detected - Checking...")
                    last_detection_time = time.time()
                    
                    # Check if it's a guyabano using color detection with more lenient thresholds
                    time.sleep(0.5)  # Give the camera a moment to stabilize
                    
                    # Get a fresh frame
                    ret, frame = cap.read()
                    if ret:
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        
                        # Define a wider range of green colors (adjust these values as needed)
                        # Lower bound (darker green)
                        lower_green1 = np.array([30, 40, 40])  # More lenient hue range
                        upper_green1 = np.array([90, 255, 255])
                        
                        # Upper bound (lighter green/yellow)
                        lower_green2 = np.array([20, 30, 30])
                        upper_green2 = np.array([100, 255, 255])
                        
                        # Create masks for both ranges and combine them
                        mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
                        mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
                        mask = cv2.bitwise_or(mask1, mask2)
                        
                        # Apply some morphological operations to clean up the mask
                        kernel = np.ones((5,5), np.uint8)
                        mask = cv2.erode(mask, kernel, iterations=1)
                        mask = cv2.dilate(mask, kernel, iterations=2)
                        
                        # Debug: Show the mask (uncomment for debugging)
                        # cv2.imshow('Color Mask', mask)
                        # cv2.waitKey(1)
                        
                        # Find contours
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Be more lenient with contour area and aspect ratio
                        is_guyabano = False
                        for c in contours:
                            area = cv2.contourArea(c)
                            if area > 30:  # Very lenient area threshold
                                x, y, w, h = cv2.boundingRect(c)
                                aspect_ratio = float(w)/h if h > 0 else 0
                                # Accept a very wide range of aspect ratios
                                if 0.2 < aspect_ratio < 5.0:
                                    is_guyabano = True
                                    break
                    else:
                        # If we couldn't get a frame, assume it's not a guyabano to keep things moving
                        is_guyabano = False
                    
                    if is_guyabano:
                        status_label.config(text="Status: Guyabano detected - Analyzing quality...")
                        servo_started = True
                        scanning_complete = False
                        Thread(target=rotate_servo_step_by_step).start()
                    else:
                        status_label.config(text="Status: No guyabano detected. Continuing...")
                        # Small delay before restarting motor to prevent rapid toggling
                        time.sleep(1)
                        motor_forward(speed=70)
                else:
                    # Only move forward if we're not already moving
                    if not motor_running:
                        motor_forward(speed=70)
            # If we're in the middle of a scan, ensure motor is stopped
            elif servo_started and not scanning_complete:
                motor_stop()
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
    if program_running:
        video_label.after(50, update_video_feed)

# ------------------ BUTTON CALLBACKS ------------------ #
def start_detection():
    global program_running, last_detection_time, start_button, stop_button, status_label
    if not program_running:
        program_running = True
        last_detection_time = time.time()
        if start_button and stop_button and status_label:
            start_button.config(state=tk.DISABLED)
            stop_button.config(state=tk.NORMAL)
            status_label.config(text="Status: Detection started")
        update_video_feed()

def stop_detection():
    global program_running
    program_running = False
    if start_button and stop_button and status_label:
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)
        status_label.config(text="Status: Stopped")
    motor_stop()

def on_reset():
    global servo_started, scanning_complete, last_detection_time, motor_running, last_scan_time
    servo_started = False
    scanning_complete = False
    last_detection_time = None
    last_scan_time = 0  # Reset cooldown timer
    motor_stop()
    set_angle(0)
    GPIO.output(relay_pin, GPIO.LOW)
    if start_button and stop_button and status_label:
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)
        status_label.config(text="Status: Ready")
    # Ensure motor starts moving if no object is detected
    dist = get_distance()
    if dist is not None and (dist > 36 and dist < 140):
        motor_forward(speed=70)

def cleanup():
    global program_running, cap
    program_running = False
    motor_stop()
    try:
        if pwm:
            pwm.ChangeDutyCycle(0)
            pwm.stop()
        if motor_pwm:
            motor_pwm.ChangeDutyCycle(0)
            motor_pwm.stop()
        if cap is not None and cap.isOpened():
            cap.release()
    except Exception as e:
        print(f"Error during cleanup: {e}")
    GPIO.cleanup()
    if 'root' in globals() and root is not None:
        root.quit()

def on_closing():
    cleanup()
    if 'root' in globals() and root is not None:
        root.destroy()

# ------------------ MAIN APPLICATION ------------------ #
if __name__ == "__main__":
    try:
        model = load_model('model.h5') if TENSORFLOW_AVAILABLE and os.path.exists('model.h5') else None
        if model is None and TENSORFLOW_AVAILABLE:
            print("Warning: Could not load model. Using color-based detection only.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            sys.exit(1)
        root = tk.Tk()
        root.title("Guyabano Detection System")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = min(800, int(screen_width * 0.9))
        window_height = min(600, int(screen_height * 0.9))
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        root.minsize(640, 480)
        main_frame = ttk.Frame(root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=0)
        main_frame.rowconfigure(2, weight=0)
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="2")
        video_frame.grid(row=0, column=0, sticky='nsew', padx=2, pady=(2, 5))
        video_label = ttk.Label(video_frame)
        video_label.pack(fill=tk.BOTH, expand=True)
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, sticky='ew', padx=2, pady=2)
        start_button = ttk.Button(button_frame, text="Start", command=start_detection, width=10)
        start_button.pack(side=tk.LEFT, padx=5, pady=2)
        stop_button = ttk.Button(button_frame, text="Stop", command=stop_detection, state=tk.DISABLED, width=10)
        stop_button.pack(side=tk.LEFT, padx=5, pady=2)
        reset_button = ttk.Button(button_frame, text="Reset", command=on_reset, width=10)
        reset_button.pack(side=tk.LEFT, padx=5, pady=2)
        status_frame = ttk.Frame(main_frame, height=24, relief=tk.SUNKEN)
        status_frame.grid(row=2, column=0, sticky='ew', padx=2, pady=(0, 2))
        status_frame.grid_propagate(False)
        distance_label = ttk.Label(status_frame, text="Distance: -- cm", padding=(5, 2))
        distance_label.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Separator(status_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        status_label = ttk.Label(status_frame, text="Status: Ready", padding=(5, 2))
        status_label.pack(side=tk.LEFT, fill=tk.Y)
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cleanup()


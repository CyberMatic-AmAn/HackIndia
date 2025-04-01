import cv2
import numpy as np
import time
from flask import Flask, Response, jsonify, render_template
import threading

# Initialize Flask app
app = Flask(__name__)

# Initialize camera
cap = cv2.VideoCapture('video_traffic.mp4')
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Global variables
vehicle_count = [0, 0, 0, 0]  # Per lane
frame_count = 0
start_time = time.time()
traffic_density_history = [[], [], [], []]  # Per lane
traffic_light_states = ["Green", "Green", "Green", "Green"]  # Per lane
current_frame = None
lock = threading.Lock()

# Background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

def data_collection():
    """Capture and return a frame from the camera."""
    ret, frame = cap.read()
    return frame if ret else None

def split_frame(frame):
    """Split frame into 4 quadrants (lanes)."""
    if frame is None:
        return [None] * 4
    height, width = frame.shape[:2]
    half_h, half_w = height // 2, width // 2
    return [
        frame[0:half_h, 0:half_w],        # Top-left (Lane 1)
        frame[0:half_h, half_w:width],    # Top-right (Lane 2)
        frame[half_h:height, 0:half_w],   # Bottom-left (Lane 3)
        frame[half_h:height, half_w:width] # Bottom-right (Lane 4)
    ]

def data_preprocessing(frame):
    """Preprocess the frame for motion detection."""
    if frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_emergency_vehicle(frame):
    """Detect emergency vehicles based on red/blue colors."""
    if frame is None:
        return False
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red color range (e.g., sirens)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    red_mask = mask1 + mask2
    # Blue color range (e.g., police/ambulance lights)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Check if significant red or blue is present
    return cv2.countNonZero(red_mask) > 500 or cv2.countNonZero(blue_mask) > 500

def traffic_pattern_analysis(frame):
    """Analyze traffic in 4 lanes and detect emergency vehicles."""
    global vehicle_count, frame_count, traffic_density_history
    if frame is None:
        return frame

    lanes = split_frame(frame)
    preprocessed_lanes = [data_preprocessing(lane) for lane in lanes]
    lane_vehicle_counts = [0] * 4
    emergency_detected = [False] * 4

    for i, (lane, preprocessed) in enumerate(zip(lanes, preprocessed_lanes)):
        if preprocessed is None:
            continue
        fgmask = fgbg.apply(preprocessed)
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 300:  # Smaller threshold for smaller quadrants
                lane_vehicle_counts[i] += 1
                x, y, w, h = cv2.boundingRect(contour)
                # Adjust coordinates based on lane position
                x_offset = 0 if i in [0, 2] else frame.shape[1] // 2
                y_offset = 0 if i in [0, 1] else frame.shape[0] // 2
                cv2.rectangle(frame, (x + x_offset, y + y_offset), 
                             (x + x_offset + w, y + y_offset + h), (0, 255, 0), 2)

        emergency_detected[i] = detect_emergency_vehicle(lane)
        vehicle_count[i] += lane_vehicle_counts[i]
        with lock:
            traffic_density_history[i].append(lane_vehicle_counts[i])

    frame_count += 1
    # Draw lane labels and lights
    for i, (count, light) in enumerate(zip(lane_vehicle_counts, traffic_light_states)):
        x, y = (10 if i in [0, 2] else frame.shape[1] // 2 + 10), (30 if i in [0, 1] else frame.shape[0] // 2 + 30)
        cv2.putText(frame, f"Lane {i+1}: {count}", (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Light: {light}", (x, y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255 if light == "Green" else 0), 2)

    return frame

def predict_traffic_and_adjust_lights():
    """Predict traffic load and adjust lights with emergency priority."""
    global traffic_density_history, traffic_light_states
    while True:
        time.sleep(5)
        with lock:
            if not all(len(history) >= 5 for history in traffic_density_history):
                continue
            for i in range(4):
                recent_densities = traffic_density_history[i][-5:]
                avg_density = sum(recent_densities) / len(recent_densities)
                emergency = detect_emergency_vehicle(split_frame(current_frame)[i] if current_frame is not None else None)

                if emergency:
                    traffic_light_states[i] = "Green"  # Priority for emergency
                    print(f"Lane {i+1}: Emergency detected, Light: Green")
                elif avg_density > 2:  # Heavy traffic
                    traffic_light_states[i] = "Red"
                elif avg_density > 0.5:  # Moderate traffic
                    traffic_light_states[i] = "Yellow"
                else:  # Light traffic
                    traffic_light_states[i] = "Green"
                print(f"Lane {i+1}: Avg Density: {avg_density:.2f}, Light: {traffic_light_states[i]}")

def generate_frames():
    """Generate MJPEG stream for video feed."""
    global current_frame
    while True:
        frame = data_collection()
        if frame is None:
            time.sleep(1)
            continue

        current_frame = traffic_pattern_analysis(frame)
        if current_frame is None:
            continue

        ret, buffer = cv2.imencode('.jpg', current_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Serve the HTML page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/traffic_data')
def traffic_data():
    """Return traffic metrics and light states as JSON."""
    global vehicle_count, frame_count, traffic_density_history, start_time, traffic_light_states
    current_time = time.time()
    with lock:
        avg_densities = [
            sum(history[-frame_count:]) / max(frame_count, 1) if frame_count > 0 else 0
            for history in traffic_density_history
        ] if current_time - start_time >= 10 else [
            sum(history[-frame_count:]) / max(frame_count, 1) if frame_count > 0 else 0
            for history in traffic_density_history
        ]
        if current_time - start_time >= 10:
            start_time = current_time
            frame_count = 0

        current_densities = [history[-1] if history else 0 for history in traffic_density_history]

    data = {
        'total_vehicles': [int(v) for v in vehicle_count],
        'current_density': current_densities,
        'avg_density': [round(d, 2) for d in avg_densities],
        'traffic_lights': traffic_light_states,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
    }
    return jsonify(data)

if __name__ == "__main__":
    print("Starting AI-based traffic system with lane-specific lights...")
    threading.Thread(target=predict_traffic_and_adjust_lights, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
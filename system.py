import cv2
import numpy as np
import time
from flask import Flask, Response, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv3-tiny model
net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Global variables
vehicle_count = 0
frame_count = 0
start_time = time.time()
traffic_density_history = []
current_frame = None

def data_collection():
    """Capture and return a frame from the camera."""
    ret, frame = cap.read()
    if ret:
        return frame
    return None

def data_preprocessing(frame):
    """Preprocess the frame for YOLO detection."""
    # Resize and prepare blob for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    return blob

def traffic_pattern_analysis(frame, blob):
    """Detect vehicles and analyze traffic patterns using YOLO."""
    global vehicle_count, frame_count, traffic_density_history

    height, width = frame.shape[:2]
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Process YOLO outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3 and classes[class_id] == 'car':  # Detect only cars
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    current_vehicle_count = len(indexes)

    # Draw bounding boxes
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"Car: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update traffic metrics
    vehicle_count += current_vehicle_count
    frame_count += 1
    density = current_vehicle_count
    traffic_density_history.append(density)

    # Display metrics on frame
    cv2.putText(frame, f"Vehicles: {current_vehicle_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Density: {density}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

def generate_frames():
    """Generate MJPEG stream for video feed."""
    global current_frame
    while True:
        frame = data_collection()
        if frame is None:
            continue

        blob = data_preprocessing(frame)
        current_frame = traffic_pattern_analysis(frame, blob)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', current_frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in MJPEG format
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
    """Return traffic metrics as JSON."""
    global vehicle_count, frame_count, traffic_density_history, start_time
    current_time = time.time()
    if current_time - start_time >= 10:
        avg_density = sum(traffic_density_history[-frame_count:]) / max(frame_count, 1)
        start_time = current_time
        frame_count = 0
    else:
        avg_density = sum(traffic_density_history[-frame_count:]) / max(frame_count, 1) if frame_count > 0 else 0

    data = {
        'total_vehicles': vehicle_count,
        'current_density': traffic_density_history[-1] if traffic_density_history else 0,
        'avg_density': avg_density,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
    }
    return jsonify(data)

if __name__ == "__main__":
    print("Starting AI-based traffic system with YOLO and web interface...")
    app.run(host='0.0.0.0', port=5000, threaded=True)
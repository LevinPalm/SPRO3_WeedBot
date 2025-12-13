import cv2
import time
import datetime
from ultralytics import YOLO
from gpiozero import PWMLED, DigitalOutputDevice
import sys
import threading
from flask import Flask, request, jsonify, render_template_string, Response 
import json 
import os   

# Constants and Configuration
MODEL_PATH = 'path/to/best_ncnn_model/'
VIDEO_SOURCE = 0 
CONFIG_FILE_PATH = 'config.json'

# Default configuration values
DEFAULT_CONFIG = {
    "motor_speed": 0.2,
    "pump_detection_duration": 0.2,
    "water_per_spray_ml": 10.0,        # Volume of water sprayed per pump activation, not measured
    "water_tank_capacity_ml": 200.0, # Total capacity of the water tank, not measured
    "current_water_level_ml": 200.0, # Last known water level
    "detection_cooldown_s": 1.0,      # Time to ignore detections after a successful spray
    "motor_pause_on_detection_s": 1.0 # How long to pause motor when a weed is detected
}

# GPIO Pin Configuration (BCM Numbers)
GPIO_DIR_PIN_MOTOR = 27 
GPIO_PWM_PIN_MOTOR = 13 
GPIO_DIR_PIN_PUMP = 24 
GPIO_PWM_PIN_PUMP = 12 
SPEED_PUMP_DUTY_CYCLE = 0.50  # Constant power for the pump

# Flask Server Config
HOST = '0.0.0.0' 
PORT = 5000       

# Global State Management Class
class WeedBotState:
    def __init__(self):
        # Motor/Pump State
        self.motor_speed = DEFAULT_CONFIG["motor_speed"]
        self.pump_detection_duration = DEFAULT_CONFIG["pump_detection_duration"]
        self.pump_off_time = 0.0 # Timestamp when pump should turn off
        
        # Water/Logging State
        self.water_per_spray_ml = DEFAULT_CONFIG["water_per_spray_ml"]
        self.water_tank_capacity_ml = DEFAULT_CONFIG["water_tank_capacity_ml"]
        self.current_water_level_ml = DEFAULT_CONFIG["current_water_level_ml"]
        self.water_log = [] # Stores log entries
        
        # Cooldown State
        self.detection_cooldown_s = DEFAULT_CONFIG["detection_cooldown_s"]
        self.last_detection_time = 0.0 # Timestamp of the last successful spray
        
        # Frame Buffer for Video Streaming
        self.latest_annotated_frame = None
        self.latest_frame_time = 0.0 
        
        # Threading Lock to protect shared state access
        self.lock = threading.RLock()
        # Motor pause state
        self.motor_paused_until = 0.0
        self.motor_paused_prev_speed = None

# Instantiate the global state object
STATE = WeedBotState()
app = Flask(__name__) # Initialize Flask application


# Configuration Persistence Functions
def save_config():
    # Saves current configuration and persistent state to a JSON file
    config = {
        "motor_speed": STATE.motor_speed,
        "pump_detection_duration": STATE.pump_detection_duration,
        "water_per_spray_ml": STATE.water_per_spray_ml,
        "water_tank_capacity_ml": STATE.water_tank_capacity_ml,
        "current_water_level_ml": STATE.current_water_level_ml,
        "detection_cooldown_s": STATE.detection_cooldown_s,
        "motor_pause_on_detection_s": STATE.motor_pause_on_detection_s
    }
    try:
        with open(CONFIG_FILE_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        print("Configuration saved successfully.")
    except Exception as e:
        print(f"Error saving config: {e}")

def load_config():
    # Loads configuration from a JSON file and updates the STATE object
    with STATE.lock: # Ensure thread safety during config loading
        try:
            if os.path.exists(CONFIG_FILE_PATH):
                with open(CONFIG_FILE_PATH, 'r') as f:
                    config = json.load(f)
                    STATE.motor_speed = config.get("motor_speed", DEFAULT_CONFIG["motor_speed"])
                    STATE.pump_detection_duration = config.get("pump_detection_duration", DEFAULT_CONFIG["pump_detection_duration"])
                    STATE.water_per_spray_ml = config.get("water_per_spray_ml", DEFAULT_CONFIG["water_per_spray_ml"])
                    STATE.water_tank_capacity_ml = config.get("water_tank_capacity_ml", DEFAULT_CONFIG["water_tank_capacity_ml"])
                    STATE.current_water_level_ml = config.get("current_water_level_ml", DEFAULT_CONFIG["water_tank_capacity_ml"])
                    STATE.detection_cooldown_s = config.get("detection_cooldown_s", DEFAULT_CONFIG["detection_cooldown_s"]) 
                    STATE.motor_pause_on_detection_s = config.get("motor_pause_on_detection_s", DEFAULT_CONFIG["motor_pause_on_detection_s"]) 
                    print(f"Configuration loaded from {CONFIG_FILE_PATH}.")
            else:
                print(f"Config file ({CONFIG_FILE_PATH}) not found. Using defaults and saving new file.")
                STATE.current_water_level_ml = DEFAULT_CONFIG["water_tank_capacity_ml"]
                save_config() 
        except Exception as e:
            print(f"Error loading config, using defaults: {e}")
            STATE.__init__() # Reset all state to defaults
            save_config()


# Load configuration at script start
load_config()


# Initialize GPIO Zero Devices
try:
    # Motor Setup
    motor_dir = DigitalOutputDevice(GPIO_DIR_PIN_MOTOR)
    motor_pwm = PWMLED(GPIO_PWM_PIN_MOTOR)
    
    # Pump Setup
    pump_dir = DigitalOutputDevice(GPIO_DIR_PIN_PUMP)
    pump_pwm = PWMLED(GPIO_PWM_PIN_PUMP)
    
    # Motor is initialized to 0 speed
    motor_dir.on() 
    motor_pwm.value = 0.0 
    
    # Update shared state and save 0.0 speed to config for clean restarts
    with STATE.lock:
        STATE.motor_speed = 0.0 # Override loaded config speed to 0
        save_config() 

    print("Motor initialized to 0 speed and state saved to config.")

    # Pump starts OFF
    pump_dir.off()
    pump_pwm.value = 0.0
    print(f"Pump initialized to OFF, detection duration set to {STATE.pump_detection_duration:.1f}s (Loaded from config).")

except Exception as e:
    print(f"Error initializing GPIO Zero devices: {e}")
    sys.exit(1)


# Thread-Safe Control Functions
def set_motor_speed(new_speed):
    # Updates the motor speed
    with STATE.lock:
        try:
            new_speed = max(0.0, min(1.0, float(new_speed)))
            STATE.motor_speed = new_speed
            motor_pwm.value = STATE.motor_speed
            print(f"Motor speed updated to {STATE.motor_speed*100:.0f}%")
            # User-initiated speed change cancels any temporary motor pause
            STATE.motor_paused_prev_speed = None
            STATE.motor_paused_until = 0.0
            save_config() 
            return True
        except ValueError:
            return False

def log_water_usage(duration, type):
    # Logs the pump activation and updates the water level
    amount_sprayed = STATE.water_per_spray_ml
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Only log and use water if the level is above 0
    if STATE.current_water_level_ml >= amount_sprayed:
        STATE.current_water_level_ml -= amount_sprayed
        STATE.water_log.append({
            'time': current_time, 
            'duration': duration, 
            'amount': amount_sprayed, 
            'type': type
        })
        STATE.water_log = STATE.water_log[-50:] 
        save_config() 
        print(f"Water Logged: {amount_sprayed:.1f}ml used. Remaining: {STATE.current_water_level_ml:.1f}ml.")
        return True
    else:
        print("Tank empty: Cannot spray water.")
        return False


# Flask Web Server Logic

# Video Stream Generator
def generate_frames():
    # Function that streams the latest annotated frame
    while True:
        frame_to_stream = None
        with STATE.lock:
            frame_to_stream = STATE.latest_annotated_frame
        
        if frame_to_stream is not None:
            ret, buffer = cv2.imencode('.jpg', frame_to_stream)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    # Video streaming route
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=['GET'])
def index():
    # Main page with GUI controls and video feed
    html_page = """
    <!doctype html>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <title>WeedBot Control</title>
    <style>
        body{
            font-family: 'Inter', sans-serif;
        }
        fieldset {
            border: 1px solid #e2e8f0; 
            border-radius: 0.5rem; 
            padding: 1rem; 
            margin-bottom: 1rem;
            background-color: white;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        legend {
            font-weight: 600;
            padding: 0 0.5rem;
            color: #1e40af;
        }
        input, button {
            transition: all 0.2s;
            border-radius: 0.375rem;
        }
        .start-btn { background-color: #10b981; color: white; }
        .stop-btn { background-color: #ef4444; color: white; }
        .test-btn { background-color: #3b82f6; color: white; }
        .refill-btn { background-color: #f97316; color: white; }
        .video-container {
            position: relative;
            padding-bottom: 75%; 
            height: 0;
            overflow: hidden;
            border-radius: 0.75rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        .video-container img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
    </style>
    <div class="min-h-screen bg-gray-100 p-4 md:p-8">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">WeedBot Control Dashboard</h1>
        
        <div class="flex flex-wrap lg:flex-nowrap gap-6 max-w-7xl mx-auto">
            
            <!-- Video Feed Section -->
            <div class="w-full lg:w-1/2">
                <h2 class="text-xl font-semibold mb-2 text-gray-700">Live Camera Feed</h2>
                <div class="video-container">
                    <img id="videoStream" src="{{ url_for('video_feed') }}" alt="Live Video Feed" onerror="this.src='https://placehold.co/600x450/f0f0f0/666?text=Camera+Stream+Offline'">
                </div>
            </div>
            
            <!-- Controls and Status Section -->
            <div class="w-full lg:w-1/2 flex flex-col gap-4">
                
                <!-- Motor Controls -->
                <fieldset>
                    <legend>Motor Control</legend>
                    <p class="mb-3">Current Motor Speed: <span id="currentSpeed" class="font-bold text-green-600">{{ speed }}%</span></p>
                    <div class="flex flex-wrap gap-2 mb-4">
                        <button onclick="startMotor()" class="start-btn py-2 px-4 shadow-md">Start Motor (20%)</button>
                        <button onclick="stopMotor()" class="stop-btn py-2 px-4 shadow-md">Stop Motor (0%)</button>
                    </div>
                    
                    <form id="speed-form" onsubmit="return setSpeed()" class="flex flex-col sm:flex-row gap-2 items-center">
                        <label for="speed_input" class="text-sm font-medium whitespace-nowrap">Speed (0.0 to 1.0):</label>
                        <input type="number" step="0.01" min="0" max="1" id="speed_input" name="speed" value="{{ speed / 100 }}" required class="border p-2 w-full sm:w-auto focus:border-blue-500">
                        <button type="submit" class="test-btn py-2 px-4 w-full sm:w-auto shadow-md">Set Speed</button>
                    </form>
                </fieldset>

                <!-- Cooldown Configuration -->
                <fieldset>
                    <legend>Detection Cooldown</legend>
                    <p class="mb-3">Current Cooldown: <span id="currentCooldown" class="font-bold text-blue-600">{{ detection_cooldown_s }}s</span></p>
                    <form id="cooldown-form" onsubmit="return setCooldown()" class="flex flex-col sm:flex-row gap-2 items-center">
                        <label for="cooldown_input" class="text-sm font-medium whitespace-nowrap">Ignore Detections For (seconds):</label>
                        <input type="number" step="0.1" min="0.5" max="30" id="cooldown_input" name="cooldown" value="{{ detection_cooldown_s }}" required class="border p-2 w-full sm:w-auto focus:border-blue-500">
                        <button type="submit" class="start-btn py-2 px-4 w-full sm:w-auto shadow-md">Set Cooldown</button>
                    </form>
                    <small class="text-gray-500 mt-1 block">Prevents spraying the same spot multiple times.</small>
                </fieldset>

                <!-- Pump Control and Duration -->
                <fieldset>
                    <legend>Pump Configuration</legend>
                    
                    <form id="pump-duration-form" onsubmit="return setDetectionDuration()" class="flex flex-col sm:flex-row gap-2 items-center mb-4">
                        <label for="detection_duration_input" class="text-sm font-medium whitespace-nowrap">Auto Spray Duration (s, max 5.0):</label>
                        <input type="number" step="0.1" min="0.1" max="5.0" id="detection_duration_input" name="duration" value="{{ detection_duration }}" required class="border p-2 w-full sm:w-auto">
                        <button type="submit" class="start-btn py-2 px-4 w-full sm:w-auto shadow-md">Set Auto Duration</button>
                    </form>

                    <form id="pump-test-form" onsubmit="return testPump()" class="flex flex-col sm:flex-row gap-2 items-center">
                        <label for="pump_duration" class="text-sm font-medium whitespace-nowrap">Test Duration (s, max 10.0):</label>
                        <input type="number" step="0.1" min="0.1" max="10" id="pump_duration" name="duration" value="1.0" required class="border p-2 w-full sm:w-auto">
                        <button type="submit" class="test-btn py-2 px-4 w-full sm:w-auto shadow-md">Test Pump</button>
                    </form>
                </fieldset>

                <!-- Water Status and Configuration -->
                <fieldset>
                    <legend>Water Tank Status</legend>
                    <p class="text-lg font-medium mb-3">
                        Remaining: <span id="waterLevel" class="text-blue-600 font-bold">{{ current_water_level_ml }}ml</span> / 
                        <span id="waterCapacity">{{ water_tank_capacity_ml }}ml</span>
                        <span id="refillStatus" class="ml-3 px-2 py-0.5 rounded text-sm font-semibold"></span>
                    </p>
                    
                    <button onclick="resetWaterLevel()" class="refill-btn py-2 px-4 shadow-md mb-4">Refill Tank (Reset Level)</button>

                    <form id="water-config-form" onsubmit="return setWaterConfig()" class="flex flex-col gap-2">
                        <p class="font-semibold text-sm mb-1 text-gray-700">Water Settings:</p>
                        <div class="flex flex-wrap gap-4">
                            <label class="text-sm">Spray per Activation (ml):</label>
                            <input type="number" step="0.1" min="0.1" id="spray_ml_input" value="{{ water_per_spray_ml }}" required class="border p-1 w-20">
                            
                            <label class="text-sm">Tank Capacity (ml):</label>
                            <input type="number" step="1" min="100" id="capacity_ml_input" value="{{ water_tank_capacity_ml }}" required class="border p-1 w-24">
                        </div>
                        <button type="submit" class="start-btn py-2 px-4 mt-2 shadow-md">Update Water Config</button>
                    </form>
                </fieldset>
            </div>
        </div>

        <!-- Log Section -->
        <div class="max-w-7xl mx-auto mt-6">
            <h2 class="text-xl font-semibold text-gray-700 mb-2">Pump Activation Log (Last 50)</h2>
            <div class="bg-white p-4 rounded-lg shadow-lg">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead>
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Duration (s)</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Amount (ml)</th>
                        </tr>
                    </thead>
                    <tbody id="waterLogTableBody" class="bg-white divide-y divide-gray-200">
                        <!-- Log entries will be inserted here by JavaScript -->
                    </tbody>
                </table>
                <p id="logStatus" class="text-center py-4 text-gray-500">Loading log...</p>
            </div>
        </div>

        <div id="status" class="mt-6 p-3 rounded-lg text-center font-medium bg-blue-100 text-blue-800 max-w-7xl mx-auto shadow-inner">Ready.</div>
    </div>
    
    <script>
        const statusDiv = document.getElementById('status');
        const waterLevelSpan = document.getElementById('waterLevel');
        const waterCapacitySpan = document.getElementById('waterCapacity');
        const refillStatusSpan = document.getElementById('refillStatus');
        const logTableBody = document.getElementById('waterLogTableBody');
        const logStatus = document.getElementById('logStatus');
        const currentCooldownSpan = document.getElementById('currentCooldown');

        document.addEventListener('DOMContentLoaded', updateWaterStatusUI);

        // Helper function for AJAX calls
        function sendCommand(endpoint, data, successMessage) {
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                xhr.open("POST", endpoint, true);
                xhr.setRequestHeader("Content-Type", "application/json");
                
                statusDiv.innerText = 'Sending command to ' + endpoint + '...';
                statusDiv.className = 'mt-6 p-3 rounded-lg text-center font-medium bg-yellow-100 text-yellow-800 max-w-7xl mx-auto shadow-inner';

                xhr.onreadystatechange = function () {
                    if (xhr.readyState === 4) {
                        if (xhr.status === 200) {
                            const response = JSON.parse(xhr.responseText);
                            statusDiv.className = 'mt-6 p-3 rounded-lg text-center font-medium bg-green-100 text-green-800 max-w-7xl mx-auto shadow-inner';
                            statusDiv.innerText = 'Status: ' + (successMessage || response.message);
                            
                            if(response.new_speed !== undefined) {
                                document.getElementById('currentSpeed').innerText = (response.new_speed * 100).toFixed(0) + '%';
                                document.getElementById('speed_input').value = response.new_speed;
                            }
                            if(response.new_duration !== undefined) {
                                document.getElementById('detection_duration_input').value = response.new_duration;
                            }
                            if(response.new_cooldown !== undefined) { // NEW: Update cooldown display
                                currentCooldownSpan.innerText = response.new_cooldown.toFixed(1) + 's';
                                document.getElementById('cooldown_input').value = response.new_cooldown;
                            }
                            
                            updateWaterStatusUI(); 
                            resolve(response);
                        } else {
                            statusDiv.className = 'mt-6 p-3 rounded-lg text-center font-medium bg-red-100 text-red-800 max-w-7xl mx-auto shadow-inner';
                            statusDiv.innerText = 'Error: ' + xhr.status + ' (' + (xhr.responseText || 'Server error') + ')';
                            reject(new Error(xhr.responseText));
                        }
                    }
                };
                xhr.send(JSON.stringify(data));
            });
        }
        
        // Motor Control Functions
        function setSpeed() {
            const speed = document.getElementById('speed_input').value;
            sendCommand('/set_speed', {speed: parseFloat(speed)});
            return false;
        }

        function startMotor() {
            sendCommand('/start_motor', {});
        }

        function stopMotor() {
            sendCommand('/stop_motor', {});
        }

        // Pump Control Functions
        function testPump() {
            const duration = document.getElementById('pump_duration').value;
            sendCommand('/test_pump', {duration: parseFloat(duration)});
            return false;
        }

        function setDetectionDuration() {
            const duration = document.getElementById('detection_duration_input').value;
            sendCommand('/set_detection_duration', {duration: parseFloat(duration)});
            return false;
        }

        // Cooldown Control Function
        function setCooldown() {
            const cooldown = document.getElementById('cooldown_input').value;
            sendCommand('/set_cooldown', {cooldown: parseFloat(cooldown)});
            return false;
        }

        // Water Management Functions
        function resetWaterLevel() {
            if (confirm('Are you sure you want to reset the water level to full capacity?')) {
                sendCommand('/reset_water_level', {}, 'Tank level reset to full capacity.');
            }
        }

        function setWaterConfig() {
            const sprayMl = document.getElementById('spray_ml_input').value;
            const capacityMl = document.getElementById('capacity_ml_input').value;
            
            sendCommand('/set_water_config', {
                water_per_spray_ml: parseFloat(sprayMl),
                water_tank_capacity_ml: parseFloat(capacityMl)
            }, 'Water configuration updated.');
            return false;
        }

        // UI Update Function
        function updateWaterStatusUI() {
            fetch('/get_status')
                .then(response => response.json())
                .then(data => {
                    // Update Status Display
                    waterLevelSpan.innerText = data.current_water_level_ml.toFixed(1) + 'ml';
                    waterCapacitySpan.innerText = data.water_tank_capacity_ml.toFixed(0) + 'ml';
                    currentCooldownSpan.innerText = data.detection_cooldown_s.toFixed(1) + 's'; // Update from state
                    
                    const percent = (data.current_water_level_ml / data.water_tank_capacity_ml) * 100;
                    
                    if (percent > 20) {
                        refillStatusSpan.innerText = 'OK';
                        refillStatusSpan.className = 'ml-3 px-2 py-0.5 rounded text-sm font-semibold bg-green-200 text-green-800';
                        waterLevelSpan.classList.remove('text-red-600');
                        waterLevelSpan.classList.add('text-blue-600');
                    } else if (percent > 0) {
                        refillStatusSpan.innerText = 'LOW';
                        refillStatusSpan.className = 'ml-3 px-2 py-0.5 rounded text-sm font-semibold bg-yellow-200 text-yellow-800';
                        waterLevelSpan.classList.remove('text-blue-600');
                        waterLevelSpan.classList.add('text-red-600');
                    } else {
                        refillStatusSpan.innerText = 'EMPTY';
                        refillStatusSpan.className = 'ml-3 px-2 py-0.5 rounded text-sm font-semibold bg-red-200 text-red-800';
                        waterLevelSpan.classList.remove('text-blue-600');
                        waterLevelSpan.classList.add('text-red-600');
                    }

                    // Update Configuration Inputs
                    document.getElementById('spray_ml_input').value = data.water_per_spray_ml.toFixed(1);
                    document.getElementById('capacity_ml_input').value = data.water_tank_capacity_ml.toFixed(0);
                    document.getElementById('cooldown_input').value = data.detection_cooldown_s.toFixed(1);
                    document.getElementById('detection_duration_input').value = data.pump_detection_duration.toFixed(1);

                    // Update Log Table
                    logTableBody.innerHTML = '';
                    if (data.water_log.length === 0) {
                        logStatus.innerText = 'No pump activations logged yet.';
                        logStatus.style.display = 'block';
                    } else {
                        logStatus.style.display = 'none';
                        data.water_log.slice().reverse().forEach(entry => { // Reverse to show latest first
                            const row = logTableBody.insertRow();
                            row.className = entry.type === 'Test' ? 'bg-gray-50' : '';
                            
                            row.insertCell(0).innerText = entry.time.split(' ')[1]; // Time only
                            row.insertCell(1).innerText = entry.type;
                            row.insertCell(2).innerText = entry.duration.toFixed(1);
                            row.insertCell(3).innerText = entry.amount.toFixed(1);
                        });
                    }
                })
                .catch(error => console.error('Error fetching status:', error));
        }

        // Fetch status every 5 seconds to keep the water level display updated
        setInterval(updateWaterStatusUI, 5000);
    </script>
    """
    
    # Use STATE attributes for initial rendering
    with STATE.lock:
        return render_template_string(html_page, 
                                      speed=STATE.motor_speed * 100,
                                      detection_duration=STATE.pump_detection_duration,
                                      water_per_spray_ml=STATE.water_per_spray_ml,
                                      water_tank_capacity_ml=STATE.water_tank_capacity_ml,
                                      current_water_level_ml=STATE.current_water_level_ml,
                                      detection_cooldown_s=STATE.detection_cooldown_s)


@app.route('/set_speed', methods=['POST'])
def handle_speed_change():
    # API endpoint to receive speed commands
    data = request.get_json()
    new_speed = data.get('speed')
    
    if new_speed is not None and set_motor_speed(new_speed):
        return jsonify({
            'status': 'success', 
            'message': f'Speed set to {STATE.motor_speed*100:.0f}% (Saved)',
            'new_speed': STATE.motor_speed
        }), 200
    else:
        return jsonify({'status': 'error', 'message': 'Invalid speed value (must be 0.0 to 1.0)'}), 400

@app.route('/stop_motor', methods=['POST'])
def handle_motor_stop():
    # API endpoint to stop the motor
    if set_motor_speed(0.0):
        return jsonify({
            'status': 'success', 
            'message': 'Motor stopped (Saved).', 
            'new_speed': 0.0
        }), 200
    else:
        return jsonify({'status': 'error', 'message': 'Failed to stop motor.'}), 500

@app.route('/start_motor', methods=['POST'])
def handle_motor_start():
    # API endpoint to start the motor (speed default 0.2)
    default_start_speed = 0.2
    if set_motor_speed(default_start_speed): 
        return jsonify({
            'status': 'success', 
            'message': f'Motor started at {default_start_speed*100:.0f}% (Saved).', 
            'new_speed': default_start_speed
        }), 200
    else:
        return jsonify({'status': 'error', 'message': 'Failed to start motor.'}), 500

@app.route('/test_pump', methods=['POST'])
def handle_pump_test():
    # API endpoint to run the pump for a test duration
    data = request.get_json()
    duration = data.get('duration', 1.0) 
    
    try:
        duration = max(0.1, min(10.0, float(duration))) 
        
        with STATE.lock: 
            if log_water_usage(duration, 'Test'):
                pump_dir.on()
                pump_pwm.value = SPEED_PUMP_DUTY_CYCLE
                STATE.pump_off_time = time.time() + duration 
            else:
                 # Return failure message if the tank is empty
                return jsonify({
                    'status': 'error', 
                    'message': 'Tank is empty. Cannot run pump test.'
                }), 400
        
        print(f"-> Pump test activate for {duration:.1f} seconds.")
        
        return jsonify({
            'status': 'success', 
            'message': f'Pump test running for {duration:.1f}s.',
            'duration': duration
        }), 200
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid duration value (must be a number between 0.1 and 10.0)'}), 400

@app.route('/set_detection_duration', methods=['POST'])
def handle_detection_duration_change():
    # API endpoint to change the pump duration variable
    data = request.get_json()
    duration = data.get('duration', STATE.pump_detection_duration)
    
    try:
        duration = max(0.1, min(5.0, float(duration))) 
        
        with STATE.lock:
            STATE.pump_detection_duration = duration 
            print(f"Detection Pump Duration updated to {STATE.pump_detection_duration:.1f} seconds.")
            save_config()
        
        return jsonify({
            'status': 'success', 
            'message': f'Detection duration set to {STATE.pump_detection_duration:.1f}s (Saved).',
            'new_duration': STATE.pump_detection_duration
        }), 200
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid duration value (must be a number between 0.1 and 5.0)'}), 400

@app.route('/set_cooldown', methods=['POST'])
def handle_cooldown_change():
    # API endpoint to change the detection cooldown time
    data = request.get_json()
    cooldown = data.get('cooldown', STATE.detection_cooldown_s)
    
    try:
        # Clamp cooldown to a reasonable range
        cooldown = max(0.5, min(30.0, float(cooldown))) 
        
        with STATE.lock:
            STATE.detection_cooldown_s = cooldown 
            print(f"Detection Cooldown updated to {STATE.detection_cooldown_s:.1f} seconds.")
            save_config()
        
        return jsonify({
            'status': 'success', 
            'message': f'Cooldown set to {STATE.detection_cooldown_s:.1f}s (Saved).',
            'new_cooldown': STATE.detection_cooldown_s
        }), 200
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid cooldown value (must be a number between 0.5 and 30.0)'}), 400

# Water Management Routes

@app.route('/get_status', methods=['GET'])
def get_status():
    # API endpoint to get the current motor speed, water level, and log
    with STATE.lock:
        return jsonify({
            'motor_speed': STATE.motor_speed,
            'pump_detection_duration': STATE.pump_detection_duration,
            'water_per_spray_ml': STATE.water_per_spray_ml,
            'water_tank_capacity_ml': STATE.water_tank_capacity_ml,
            'current_water_level_ml': STATE.current_water_level_ml,
            'detection_cooldown_s': STATE.detection_cooldown_s, # Return cooldown
            'water_log': STATE.water_log
        }), 200

@app.route('/set_water_config', methods=['POST'])
def set_water_config():
    # API endpoint to change water volume settings
    data = request.get_json()
    new_spray_ml = data.get('water_per_spray_ml')
    new_capacity_ml = data.get('water_tank_capacity_ml')

    try:
        with STATE.lock:
            if new_spray_ml is not None:
                STATE.water_per_spray_ml = max(0.1, float(new_spray_ml))
            if new_capacity_ml is not None:
                new_capacity_ml = max(100.0, float(new_capacity_ml))
                if new_capacity_ml < STATE.current_water_level_ml:
                     STATE.current_water_level_ml = new_capacity_ml 
                STATE.water_tank_capacity_ml = new_capacity_ml
            
            save_config()
            return jsonify({
                'status': 'success', 
                'message': 'Water configuration updated and saved.',
                'water_per_spray_ml': STATE.water_per_spray_ml,
                'water_tank_capacity_ml': STATE.water_tank_capacity_ml
            }), 200
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid number provided for water configuration.'}), 400

@app.route('/reset_water_level', methods=['POST'])
def reset_water_level():
    # API endpoint to reset water level to full capacity
    with STATE.lock:
        STATE.current_water_level_ml = STATE.water_tank_capacity_ml
        STATE.water_log = [] # Clear old log on refill
        save_config()
        print("Tank refilled and log cleared.")
        return jsonify({
            'status': 'success', 
            'message': 'Water tank level reset to full capacity.',
            'current_water_level_ml': STATE.current_water_level_ml
        }), 200


def run_flask():
    # Starts the Flask server in the background
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)


# Pump Control Functions
def activate_pump_on_detect():
    # Activates the pump if a detection is made and the robot is not on cooldown
    with STATE.lock:
        
        # Cooldown Check
        if time.time() < STATE.last_detection_time + STATE.detection_cooldown_s:
            return

        # Activation Check (Pump duration done AND water available)
        if time.time() > STATE.pump_off_time and STATE.current_water_level_ml > 0:
            
            # Log usage (which performs the water level subtraction and saves config)
            if log_water_usage(STATE.pump_detection_duration, 'Auto'):
                
                # Physical activation and state update
                print(f"--- OBJECT DETECTED: ACTIVATING PUMP for {STATE.pump_detection_duration:.1f}s! ---")
                pump_dir.on()
                pump_pwm.value = SPEED_PUMP_DUTY_CYCLE
                
                # Set stop time and cooldown time
                STATE.pump_off_time = time.time() + STATE.pump_detection_duration
                STATE.last_detection_time = time.time() 
                # Pause motor while spraying
                try:
                    pause_duration = float(STATE.motor_pause_on_detection_s)
                except Exception:
                    pause_duration = 1.0
                pause_motor_for(pause_duration)
        
        # Empty tank warning
        elif STATE.current_water_level_ml <= 0 and time.time() > STATE.pump_off_time:
            print("Tank empty. Cannot spray.")

    
def check_and_stop_pump():
    # Checks the timestamp and turns the pump off if the duration is complete
    if time.time() > STATE.pump_off_time and STATE.pump_off_time != 0.0:
        print("Stopping Pump")
        pump_pwm.value = 0.0
        pump_dir.off() 
        STATE.pump_off_time = 0.0


def pause_motor_for(duration_s: float):
    # Temporarily pause the motor for duration_s seconds
    now = time.time()
    with STATE.lock:
        # If user is currently running the motor at some speed and it is not already paused, store it
        if STATE.motor_paused_prev_speed is None:
            try:
                # Save the current desired speed so it can be restored after pause
                prev = float(STATE.motor_speed)
            except Exception:
                prev = 0.0
            STATE.motor_paused_prev_speed = prev

        # Set motor to OFF and set the logical motor speed to 0.0 (temporary)
        try:
            # Save the previously desired speed in motor_paused_prev_speed and
            # set the current desired speed to 0 so the rest of the system sees the motor as stopped
            STATE.motor_speed = 0.0
            motor_pwm.value = 0.0
        except Exception:
            pass

        # Extend or set the paused-until timestamp
        STATE.motor_paused_until = max(STATE.motor_paused_until, now + float(duration_s))
        print(f"-> Motor paused for {duration_s:.1f}s (will resume at {STATE.motor_paused_until:.1f}).")


def check_and_resume_motor():
    # If the motor was temporarily paused for spraying and the pause time has elapsed,
    # restore the previous motor speed unless the user has changed it in the meantime.

    with STATE.lock:
        if STATE.motor_paused_prev_speed is not None and time.time() > STATE.motor_paused_until:
            # Only restore if the current stored speed is zero
            # If user manually changed speed during the pause, not override their choice
            current_saved_speed = float(STATE.motor_speed)
            if current_saved_speed == 0.0:
                restore_speed = float(STATE.motor_paused_prev_speed)
                STATE.motor_speed = restore_speed
                try:
                    motor_pwm.value = restore_speed
                except Exception:
                    pass
                print(f"-> Motor resumed to {restore_speed*100:.0f}% after pause.")

            # Clear the pause metadata
            STATE.motor_paused_prev_speed = None
            STATE.motor_paused_until = 0.0


# Initilization YOLO and CV2
try:
    # Load the YOLO model
    try:
        model = YOLO(MODEL_PATH, task='detect') 
    except FileNotFoundError:
        print(f"Error: Model not found at {MODEL_PATH}")
        raise SystemExit 

    # Open the video source
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}.")
        raise SystemExit

    # Start the flask web server thread
    server_thread = threading.Thread(target=run_flask, daemon=True)
    server_thread.start()
    print(f"Web GUI server running on http://<Pi-IP-Address>:{PORT}")

    print("Starting object detection and control loop")
    
    # Main loop
    while cap.isOpened():
        try:
            success, frame = cap.read()
            if not success:
                print("End of video stream or read error. Attempting to re-open camera...")
                cap.release()
                time.sleep(3) 
                cap = cv2.VideoCapture(VIDEO_SOURCE)
                continue
            
            # Perform detection
            results = model.predict(source=frame, verbose=False)
            
            # Check detection and control
            detection_found = False
            if results and len(results) > 0 and len(results[0].boxes) > 0:
                detection_found = True
                
            # Annotate frame and update buffer
            if detection_found:
                annotated_frame = results[0].plot() 
            else:
                annotated_frame = frame 
                
            with STATE.lock:
                STATE.latest_annotated_frame = annotated_frame 
                STATE.latest_frame_time = time.time() 

            # Control logic
            if detection_found:
                activate_pump_on_detect()
            
            check_and_stop_pump() 
            check_and_resume_motor()

            time.sleep(0.01)

        except Exception as e:
            print(f"ERROR inside main processing loop: {e}. Skipping frame and waiting 0.5s.")
            time.sleep(0.5) 
            continue 


# Exception handling and cleanup
except SystemExit:
    pass 
except KeyboardInterrupt:
    print("\nScript stopped by user (Ctrl+C).")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
    
finally:
    # Cleanup
    print("\nCleaning up resources...")
    if 'cap' in locals():
        cap.release()
    try:
        motor_pwm.value = 0.0
        pump_pwm.value = 0.0
    except NameError:
        pass
    print("Cleanup complete. Motor control script terminated.")
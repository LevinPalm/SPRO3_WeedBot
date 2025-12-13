import cv2
import time
from ultralytics import YOLO

# Path to your exported NCNN model directory
MODEL_PATH = './best_ncnn_model/'

# Set the desired display width and height for your output window
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480 

# USB webcam usually has index 0
VIDEO_SOURCE = 0 
WINDOW_TITLE = "YOLO Detection Output"

# Load the custom YOLO model
try:
    model = YOLO(MODEL_PATH) 
except FileNotFoundError:
    print(f"Error: Model not found at {MODEL_PATH}")
    exit()

# Open the video source
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Check if the video source was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video source {VIDEO_SOURCE}. Check camera index.")
    exit()

print("Starting object detection...")

# FPS Calculation 
prev_time = cv2.getTickCount()

while cap.isOpened():
    # FPS Start Measurement for the current frame
    new_time = cv2.getTickCount()
    
    # Read a frame from the video source
    success, frame = cap.read()
    
    if not success:
        # End of video or error
        print("End of video stream or read error.")
        break
    
    # Perform Detection
    # The predict method handles all preprocessing and post-processing
    results = model.predict(source=frame, verbose=False)
    
    # Console Output of Detections
    if results and len(results) > 0:
        res = results[0] # Get the Results object for the current frame
        
        print("\n--- Frame Detections ---")
        
        # Check if any objects were detected
        if len(res.boxes) > 0:
            # Iterate through all detected bounding boxes
            for box in res.boxes:
                # Get certainty (confidence score) - extract the single float value
                confidence = box.conf[0].item()
                
                # Get the position (bounding box coordinates [x1, y1, x2, y2])
                # .xyxy returns a tensor; we convert it to a list and cast to integer coordinates
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
                
                # Get the detected object name (model/class name)
                class_index = int(box.cls[0])
                class_name = res.names[class_index]

                # Output the required information to the console
                print(f"Detected Object: {class_name}")
                print(f"  Certainty: {confidence:.2f}")
                print(f"  Position (x1, y1, x2, y2): ({x1}, {y1}, {x2}, {y2})")
        else:
            print("No objects detected in this frame.")
    
    # Draw boxes on original frame
    # The .plot() function draws the bounding boxes and labels onto a copy of the frame.
    annotated_frame = results[0].plot()

    # Calculate time taken for one frame processing
    time_taken = (new_time - prev_time) / cv2.getTickFrequency()
    # Calculate FPS (Frames Per Second)
    fps = 1 / time_taken
    # Format the FPS text
    fps_text = f"FPS: {fps:.2f}"
    
    # Display the FPS on the annotated frame
    cv2.putText(
        annotated_frame, 
        fps_text, 
        (10, 30), # Position (x, y)
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, # Font scale
        (0, 255, 0), # Color (BGR: Green)
        2 # Thickness
    )
    # Update the previous time for the next iteration
    prev_time = new_time
    
    # Resize the annotated frame to desired window size
    display_frame = cv2.resize(annotated_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    
    # Display the resulting frame
    cv2.imshow(WINDOW_TITLE, display_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
print("Script finished and resources released.")
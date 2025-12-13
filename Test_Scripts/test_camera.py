import cv2

CAMERA_INDEX = 0

# Resolution
WIDTH = 640
HEIGHT = 480

# Initialize the video capture object
cap = cv2.VideoCapture(CAMERA_INDEX)

# Check if the camera was opened successfully
if not cap.isOpened():
    print(f"Error: Cannot open camera with index {CAMERA_INDEX}.")
    print("Please check if the camera is connected and the index is correct.")
    exit()

# Set the desired frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


print(f"Starting USB camera feed (Index {CAMERA_INDEX}). Press 'q' key in the video window to exit.")

try:
    while True:
        # Read a new frame
        # 'ret' is a boolean flag (True if frame was read correctly)
        # 'frame' is the actual image data (a NumPy array)
        ret, frame = cap.read()
        
        # Check if the frame was read correctly
        if not ret:
            print("Error: Failed to retrieve frame. Exiting.")
            break

        # Display the original frame
        cv2.imshow("USB Camera Feed", frame)
        
        # Check for the 'q' key press to exit
        # cv2.waitKey(1) waits 1 millisecond for a key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An unexpected error occurred: {e}")

finally:
    # Release the camera object and close all display windows
    cap.release()
    cv2.destroyAllWindows()
    print("Camera and OpenCV resources released.")
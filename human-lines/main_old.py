import cv2
import numpy as np

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open a connection to the built-in camera (camera index 0)
cap = cv2.VideoCapture(0)

# Set the frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Could not open camera")
    exit()

# Initialize a dictionary to store the previous center positions and amplitudes for each face
prev_centers = {}

# Set the maximum amplitude limit
max_amplitude = 30

# Set the moving average window size and temporal filter alpha value
window_size = 5
alpha = 0.3

# Initialize a dictionary to store the amplitude values for each face
amplitude_values = {}

# Loop to read frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If the frame was read successfully, process it
    if ret:
        # Convert the frame to grayscale for the face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw a vertical wiggling line in the center of each detected face
        for i, (x, y, w, h) in enumerate(faces):
            center_x = x + w // 2

            # Calculate the difference in position between the current and previous frame for each face
            if i in prev_centers:
                diff = np.abs(center_x - prev_centers[i]["x"])
                amplitude = diff * 5  # Scale factor to control the wiggle amplitude
                amplitude = min(amplitude, max_amplitude)  # Limit the maximum amplitude

                # Store the amplitude value in the amplitude_values dictionary
                if i not in amplitude_values:
                    amplitude_values[i] = []
                amplitude_values[i].append(amplitude)

                # Apply a moving average filter to smooth out the amplitude values
                if len(amplitude_values[i]) > window_size:
                    amplitude_values[i].pop(0)
                amplitude = np.mean(amplitude_values[i])

                # Apply a temporal filter to reduce sudden changes in amplitude
                prev_amplitude = prev_centers[i]["amplitude"]
                amplitude = alpha * amplitude + (1 - alpha) * prev_amplitude
            else:
                amplitude = 0

            # Generate a sine wave to create the wiggle effect
            wiggle = np.sin(np.linspace(0, 2 * np.pi, frame.shape[0])) * amplitude

            # Draw the wiggling line
            for row in range(frame.shape[0]):
                col = int(center_x + wiggle[row])
                col = np.clip(col, 0, frame.shape[1] - 1)  # Clip to screen boundaries
                frame[row, col] = (0, 0, 255)

            # Store the current center position and amplitude for the next frame
            prev_centers[i] = {"x": center_x, "amplitude": amplitude}

        # Display the frame with the wiggling lines
        cv2.imshow("Camera Feed", frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Could not read frame from camera")
        break

# Release the camera and destroy the window
cap.release()
cv2.destroyAllWindows()
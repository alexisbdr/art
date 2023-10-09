import cv2
import numpy as np


class FrequencyOfChange:
    def __init__(self, num_dims):
        self.num_dims = num_dims
        self.prev_coord = None
        self.freq = [0] * num_dims

    def update(self, coord):
        if self.prev_coord is None:
            self.prev_coord = coord
            return

        for d in range(self.num_dims):
            if coord[d] != self.prev_coord[d]:
                self.freq[d] += 1

        self.prev_coord = coord

    def get_frequency(self):
        if self.prev_coord is None:
            return [0] * self.num_dims

        num_updates = sum(self.freq)
        return [f / num_updates for f in self.freq]


def locate_human(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar cascade classifier for human detection
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

    # Detect human objects in the grayscale frame using the classifier
    objects = classifier.detectMultiScale(gray)

    # Check if any human objects are detected
    if len(objects) > 0:
        # Get the location of the first human object
        x, y, w, h = objects[0]

        # Return the location as a tuple of (x, y, width, height)
        return (x, y, w, h)

    # Return None if no human objects are detected
    return None


def main():
    # Create a VideoCapture object to capture frames from the camera
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error opening camera")
        exit()

    freq_change = FrequencyOfChange(number_of_dimensions=2)

    # Loop through the frames captured from the camera
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame is read successfully
        if not ret:
            print("Error reading frame")
            break
        
        loc = locate_human(frame)

        freq_change.update(loc)

        # Render a white frame
        if loc is None:
            frame = np.full_like(frame, (255, 255, 255))
        
        # Display the frame in a window
        cv2.imshow("Camera", frame)

        # Wait for a key event
        key = cv2.waitKey(1)

        # Check if the 'q' key is pressed to quit the loop
        if key == ord('q'):
            break

    # Release the VideoCapture object and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import numpy as np
import math 
import time

class HumanChange:

    def __init__(self):
        
        ## Initialize state variables
        self.prev_coord = [0,0]
        self.prev_time = cv2.getTickCount()
        self.prev_frequency = 0
        self.prev_amplitude = 0
        self.phase_shift = 0

        ## Frequency Mapping parameters
        self.max_speed = 500  
        self.min_speed = 0 
        self.max_freq = 50
        self.min_freq = 0
        self.freq_alpha = 0.1

        ## Amplitude Mapping Parameters
        self.min_distance = 0 
        self.max_distance = 100 
        self.min_amplitude = 0
        self.max_amplitude = 100 
        self.amplitude_alpha = 0.1      

    def update(self, coord):

        #First calculate the total movement
        if coord != self.prev_coord:
            self.distance = math.sqrt((coord[0] - self.prev_coord[0]) ** 2 + (coord[1] - self.prev_coord[1]) ** 2)
            curr_time = cv2.getTickCount()
            time_diff = (curr_time - self.prev_time) / cv2.getTickFrequency()
            self.prev_time = curr_time
            self.speed = self.distance / time_diff
        
        self.prev_coord = coord

    def get_frequency(self, smoothing = True):
        """Mapping from calculated speed to frequency
        --> Linear Mapping using min max
        """
        if self.speed < self.min_speed:
            frequency = self.min_freq
        elif self.speed > self.max_speed:
            frequency = self.max_freq
        else:
            # Linear interpolation
            frequency = self.min_freq + (self.max_freq - self.min_freq) * ((self.speed - self.min_speed) / (self.max_speed - self.min_speed)) 

        if smoothing:
            frequency = self.temporal_smoothing(frequency, self.prev_frequency, self.freq_alpha)

        self.prev_frequency = frequency
        
        return self.prev_frequency


    def get_amplitude(self, smoothing = True):
        """Mapping from calculated distance to amplitude
        Linear Mapping using min max
        """
        if self.distance < self.min_distance:
            amplitude = self.min_amplitude
        elif self.distance > self.max_distance:
            amplitude = self.max_amplitude
        else:
            # Linear interpolation
            amplitude = self.min_amplitude + (self.max_amplitude - self.min_amplitude) * ((self.distance - self.min_distance) / (self.max_distance - self.min_distance)) 

        if smoothing:
            amplitude = self.temporal_smoothing(amplitude, self.prev_amplitude, self.amplitude_alpha)

        self.prev_amplitude = amplitude
        
        return self.prev_amplitude


    def get_phase_shift(self):
        """Returns Phase Shift
        """
        return self.phase_shift + 0.1

    
    def temporal_smoothing(self, curr, prev, alpha):
        """Apply Temporal Smoothing 
        """
        filtered = alpha * curr + (1 - alpha) * prev
        return filtered



def generate_wiggle(start_y, end_y, baseline_x, amplitude, frequency, phase):
    wiggle_points = []
    for y in range(start_y, end_y):
        x = baseline_x + amplitude * math.sin(frequency * (y / end_y) * 2 * math.pi + phase)
        wiggle_points.append((int(x), int(y)))
    return wiggle_points


def locate_human(frame, face=True):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar cascade classifier for human detection
    if face:
        classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    else:
        classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
    
    # Detect human objects in the grayscale frame using the classifier
    objects = classifier.detectMultiScale(gray)

    # Check if any human objects are detected
    if len(objects) > 0:
        
        # Get the location of the first human object
        x, y, w, h = objects[0]

        # Return the location as a tuple of (x, y, width, height)
        return objects

    # Return None if no human objects are detected
    return None


def main():
    # Create a VideoCapture object to capture frames from the camera
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error opening camera")
        exit()

    change = HumanChange()

    # Loop through the frames captured from the camera
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame is read successfully
        if not ret:
            print("Error reading frame")
            break
        
        loc = locate_human(frame)

        frame = np.full_like(frame, (255, 255, 255))
        if loc is None:
            frame = np.full_like(frame, (255, 255, 255))
        else:
            change.update(loc)
            start_y = loc[1]
            end_y = loc[1] + loc[3]
            baseline_x = loc[0] + loc[2] / 2
            amplitude = change.get_amplitude()
            frequency = change.get_frequency()
            phase_shift = 0
            wiggle = generate_wiggle(start_y, end_y, baseline_x, amplitude, frequency, phase_shift)
            for i in range(len(wiggle) - 1):
                cv2.line(frame, wiggle[i], wiggle[i+1], (0, 20, 250), 1)
            
    
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
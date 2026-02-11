import cv2
import os

# Set font directory for Qt to avoid warning on Linux
os.environ['QT_QPA_FONTDIR'] = '/usr/share/fonts'

def main():
    # Initialize the camera (default camera is index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera opened successfully. Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the resulting frame
        cv2.imshow('SmartView - Camera Feed', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and program finished.")

if __name__ == "__main__":
    main()

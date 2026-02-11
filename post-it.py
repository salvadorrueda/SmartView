import cv2
import numpy as np
import os

# Set font directory for Qt to avoid warning on Linux
os.environ['QT_QPA_FONTDIR'] = '/usr/share/fonts'

# Global variable to store the current HSV frame for the mouse callback
hsv_frame = None

def get_color_values(event, x, y, flags, param):
    """
    Mouse callback function to print color values of the clicked pixel.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        if hsv_frame is not None:
            # Get the HSV value at the clicked coordinates
            hsv_val = hsv_frame[y, x]
            # OpenCV HSV: H [0,179], S [0,255], V [0,255]
            print(f"Clicked at ({x}, {y}) -> HSV: {hsv_val}")

def main():
    global hsv_frame
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Create window and set mouse callback
    cv2.namedWindow('SmartView - Multi-Color Post-it Detector')
    cv2.setMouseCallback('SmartView - Multi-Color Post-it Detector', get_color_values)

    print("Post-it detector started. Searching for GREEN, YELLOW, and PINK squares.")
    print("TIP: Click on any point in the image to see its HSV values in the terminal.")
    print("Press 'q' to quit.")

    # 1. Define color ranges in HSV and their display properties
    colors = {
        "GREEN": {
            "lower": np.array([35, 40, 40]),
            "upper": np.array([85, 255, 255]),
            "draw_color": (0, 255, 0)
        },
        "YELLOW": {
            "lower": np.array([20, 80, 80]),
            "upper": np.array([30, 255, 255]),
            "draw_color": (0, 255, 255)
        },
        "PINK": {
            "lower": np.array([140, 80, 80]),
            "upper": np.array([180, 255, 255]),
            "draw_color": (147, 20, 255) # Magenta/Pinkish BGR
        }
    }

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert to HSV color space once
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_frame = hsv # Store in global for the color picker
        
        # Kernel for morphological operations
        kernel = np.ones((5, 5), np.uint8)

        # 2. Iterate through each defined color
        for color_name, props in colors.items():
            # Create a mask for the current color
            mask = cv2.inRange(hsv, props["lower"], props["upper"])

            # Clean up the mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                # Filter by area to avoid noise
                if cv2.contourArea(cnt) > 2000:
                    # Shape approximation (looking for 4 corners = rectangle/square)
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

                    if len(approx) == 4:
                        # Draw a bounding box around the detected square
                        x, y, w, h = cv2.boundingRect(approx)
                        
                        # Ensure it's somewhat square-like
                        aspect_ratio = float(w)/h
                        if 0.7 <= aspect_ratio <= 1.3:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), props["draw_color"], 3)
                            cv2.putText(frame, f"{color_name} POST-IT", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, props["draw_color"], 2)

        # Display the result
        cv2.imshow('SmartView - Multi-Color Post-it Detector', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Program finished.")

if __name__ == "__main__":
    main()

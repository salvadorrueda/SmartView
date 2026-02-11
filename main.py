import cv2
import os
from ultralytics import YOLOWorld

# Set font directory for Qt to avoid warning on Linux
os.environ['QT_QPA_FONTDIR'] = '/usr/share/fonts'

def main():
    # --- MODEL CONFIGURATION ---
    # Choose your model size based on your hardware:
    # 'yolov8s-worldv2.pt' -> Small (Fastest)
    # 'yolov8m-worldv2.pt' -> Medium
    # 'yolov8l-worldv2.pt' -> Large (Better accuracy)
    # 'yolov8x-worldv2.pt' -> Extra Large (Best accuracy, slowest)
    model_size = 'yolov8l-worldv2.pt' 
    
    # Load the selected YOLO-World model
    model = YOLOWorld(model_size)

    # Define the specific objects you want to detect (e.g., "pen")
    # You can change this list to detect whatever you want!
    custom_classes = ["post-it","human"]
    model.set_classes(custom_classes)

    # Initialize the camera (default camera is index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print(f"Camera opened. Detecting ONLY: {', '.join(custom_classes)}")
    print("Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Run YOLO-World inference on the frame
        # stream=True is more efficient for video
        results = model(frame, stream=True, verbose=False)

        # Visualize the results on the frame
        # Since we set_classes, only those will be included in the results
        for r in results:
            annotated_frame = r.plot()

        # Display the resulting annotated frame
        cv2.imshow('SmartView - Custom Object Detection', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and program finished.")

if __name__ == "__main__":
    main()

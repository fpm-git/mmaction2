import cv2

# Define the RTSP stream URL
rtsp_url = 'rtsp://localhost:8554/rtsp_in'

def main():
    # Create a VideoCapture object
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"Error: Could not open stream {rtsp_url}")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Display the resulting frame
        cv2.imshow('RTSP Stream', frame)

        # Press 'q' on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
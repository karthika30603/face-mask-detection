import cv2

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect face and predict mask
def detect_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Define the region of interest (ROI) for the face
        face_roi = frame[y:y + h, x:x + w]

        # Use the face bounding box to determine mask or no mask
        mask_color = (0, 255, 0)  # Green color for "Mask"
        no_mask_color = (0, 0, 255)  # Red color for "No Mask"

        # Assuming a person is not wearing a mask if face is detected
        label = "No Mask"
        color = no_mask_color

        # Check if mouth region is within the face bounding box
        mouth_y = int(y + 0.6 * h)
        if mouth_y < y + h:
            label = "Mask"
            color = mask_color

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return frame

# Function to detect mask in a video file
def video_mask_detection(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = detect_mask(frame)

        cv2.imshow('Mask Detection', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Provide the video file directly
video_mask_detection("mask.mp4")

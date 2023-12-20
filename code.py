pip install mediapipe opencv-python numpy
pip install opencv-python==4.5.3.56
import cv2
import mediapipe as mp

# Function to recognize hand signs
def recognize_hand_sign(landmarks):
    # Define regions of interest for rock, paper, and scissors signs
    rock_region = [(4, 5, 6), (8, 12, 16), (0, 2, 3)]
    paper_region = [(8, 12, 16), (0, 2, 3), (5, 9, 13)]
    scissors_region = [(5, 9, 13), (0, 2, 3), (8, 12, 16)]

    # Check if the landmarks are in the defined regions for rock, paper, or scissors
    if all(landmarks[i][1] < landmarks[j][1] for i, j, _ in rock_region):
        return "Rock"
    elif all(landmarks[i][1] < landmarks[j][1] for i, j, _ in paper_region):
        return "Paper"
    elif all(landmarks[i][1] < landmarks[j][1] for i, j, _ in scissors_region):
        return "Scissors"
    else:
        return "Other Sign"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp.drawing_utils.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Recognize hand sign
            hand_sign = recognize_hand_sign(landmarks.landmark)

            # Display the recognized hand sign
            cv2.putText(frame, hand_sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Hand Sign Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

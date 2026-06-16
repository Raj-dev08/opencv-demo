import cv2
import mediapipe as mp
from deepface import DeepFace
print(dir(mp))


mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose

face = mp_face.FaceDetection()
pose = mp_pose.Pose()
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    # print(results.multi_hand_landmarks)

    resultForFace = face.process(frame_rgb)
    resultForPose = pose.process(frame_rgb)

    # print("face",resultForFace.detections)
    # print(resultForPose.pose_landmarks)
    
   
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, c = frame.shape

            landmarks = hand_landmarks.landmark

        # Index finger
        index_tip = landmarks[8]
        index_pip = landmarks[6]

        # Other fingers (to ensure they are down)
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        # Condition
        if ((index_tip.y < index_pip.y and   # index up
            middle_tip.y > landmarks[10].y and
            ring_tip.y > landmarks[14].y and
            pinky_tip.y > landmarks[18].y)) or ( (
                index_tip.y > index_pip.y and   # index up
                middle_tip.y < landmarks[10].y and
                ring_tip.y < landmarks[14].y and
                pinky_tip.y < landmarks[18].y)
            ):

            cv2.putText(frame, "INDEX UP", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        for id, lm in enumerate(hand_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            print(id, cx, cy)

            # draw a small circle on each point
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # landmarks
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)  # connections
        )
    
    if resultForFace.detections:
        
        for detection in resultForFace.detections:
            bbox = detection.location_data.relative_bounding_box

            h, w, c = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            face_crop = frame[y:y+height, x:x+width]

            analysis = DeepFace.analyze(img_path=face_crop, 
                        actions=['age', 'gender', 'race', 'emotion'])

       
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe
import pyautogui
face_mesh_landmarks = mediapipe.solutions.face_mesh.FaceMesh(refine_landmarks=True)
camera = cv2.VideoCapture(0)
screenWidth, screenHeight = pyautogui.size()

while True:
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)
    windowWidth, windowHeight, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh_landmarks.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    if landmark_points:
        oneFace = landmark_points[0].landmark
        for id, ldmk_points in enumerate(oneFace[474:478]):
            x = int(ldmk_points.x * frame.shape[1])
            y = int(ldmk_points.y * frame.shape[0])
            #print(x, y)
            if id == 1:
                mouseX = int(screenWidth / windowWidth * x)
                mouseY = int(screenHeight / windowHeight * y)
                pyautogui.moveTo(mouseX, mouseY)
            cv2.circle(frame, (x, y), 3, (255, 0, 0))
        leftEye = [oneFace[145],oneFace[159]]
        for ldmk_points in leftEye:
            x = int(ldmk_points.x * frame.shape[1])
            y = int(ldmk_points.y * frame.shape[0])
            #print(x, y)
            cv2.circle(frame, (x, y), 3, (0, 0, 255))
        if (leftEye[0].y - leftEye[1].y) < 0.01:
            pyautogui.click()
            pyautogui.sleep(2)
            print("Click")
    cv2.imshow("Eye Controlled Mouse", frame)
    key = cv2.waitKey(100)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()

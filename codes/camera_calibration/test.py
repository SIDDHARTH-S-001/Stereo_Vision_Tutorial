import cv2

# Try to open both cameras using DirectShow backend
cam0 = cv2.VideoCapture(0, cv2.CAP_ANY)
cam1 = cv2.VideoCapture(-1, cv2.CAP_ANY)

# Set lower resolution and FPS to avoid bandwidth issues
for cam in (cam0, cam1):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 15)

# Check if cameras are opened
if not cam0.isOpened():
    print("❌ Camera 0 failed to open.")
if not cam1.isOpened():
    print("❌ Camera 1 failed to open.")

if not cam0.isOpened() or not cam1.isOpened():
    print("⚠️ Exiting: One or both cameras could not be opened.")
    cam0.release()
    cam1.release()
    cv2.destroyAllWindows()
    exit()

# Main loop
while True:
    ret0, frame0 = cam0.read()
    ret1, frame1 = cam1.read()

    if ret0:
        cv2.imshow('Cam 0', frame0)
    else:
        print("⚠️ Failed to read from Cam 0")

    if ret1:
        cv2.imshow('Cam 1', frame1)
    else:
        print("⚠️ Failed to read from Cam 1")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam0.release()
cam1.release()
cv2.destroyAllWindows()

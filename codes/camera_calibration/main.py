import cv2

# Open both cameras (adjust indexes if needed)
cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(2)

# Set resolution and format to MJPEG
def configure_camera(cam, width=1280, height=960, fps=30):
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FPS, fps)

configure_camera(cam1)
configure_camera(cam2)

# Confirm actual format and resolution
print("Camera 1 Format:", int(cam1.get(cv2.CAP_PROP_FOURCC)))
print("Camera 2 Format:", int(cam2.get(cv2.CAP_PROP_FOURCC)))

# cv2.namedWindow('Blended Camera Feed', cv2.WINDOW_NORMAL)

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1 or not ret2:
        print("Failed to grab one or both frames")
        break

    # Resize frame2 to match frame1, just in case
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    # Blend the two frames equally
    blended_frame = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)

    # Flip horizontally for mirror effect (optional)
    # cv2.imshow('Blended Camera Feed', cv2.flip(blended_frame, 1))
    cv2.imshow('F1',frame1)
    cv2.imshow('F2',frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam1.release()
cam2.release()
cv2.destroyAllWindows()

import cv2
import time
import numpy as np
import os

def main():
    # Camera configuration
    DEVICES = ['/dev/video2', '/dev/video3']  # Try both device nodes
    WIDTH, HEIGHT = 320, 240
    FPS = 10
    MAX_ERRORS = 5
    
    # System preparation
    os.system("echo 1000 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb")
    os.system("sudo modprobe -r uvcvideo")
    os.system("sudo modprobe uvcvideo quirks=0x80 nodrop=1")
    
    current_device = 0
    cap = None
    last_valid = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    while True:
        # Initialize or reinitialize camera
        if cap is None or not cap.isOpened():
            device = DEVICES[current_device]
            print(f"Trying {device}...")
            
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if cap.isOpened():
                print(f"Success with {device}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, FPS)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                error_count = 0
            else:
                print(f"Failed with {device}")
                current_device = (current_device + 1) % len(DEVICES)
                time.sleep(1)
                continue
        
        # Capture frame
        ret, frame = cap.read()
        
        if not ret or frame is None or frame.size == 0:
            error_count += 1
            print(f"Error {error_count}/{MAX_ERRORS}")
            frame = last_valid
            
            if error_count >= MAX_ERRORS:
                print("Releasing camera...")
                cap.release()
                cap = None
                current_device = (current_device + 1) % len(DEVICES)
                time.sleep(1)
                continue
        else:
            error_count = 0
            last_valid = frame.copy()
        
        # Display frame
        cv2.imshow(f'CAM2 ({WIDTH}x{HEIGHT} @ {FPS}fps)', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Cleanup
    if cap and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
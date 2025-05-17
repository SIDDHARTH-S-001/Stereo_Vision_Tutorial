import cv2
import time
import numpy as np
import os

def reset_usb_port(port_id="2.1.4"):
    try:
        os.system(f"echo '2-{port_id}' | sudo tee /sys/bus/usb/drivers/usb/unbind")
        time.sleep(2)
        os.system(f"echo '2-{port_id}' | sudo tee /sys/bus/usb/drivers/usb/bind")
        time.sleep(3)
        return True
    except:
        return False

def main():
    # Camera config
    DEVICE = '/dev/video0'
    WIDTH, HEIGHT = 320, 240
    FPS = 10
    
    while True:
        print(f"Initializing {DEVICE}...")
        cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
        
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, FPS)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            
            last_valid = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            error_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    error_count += 1
                    print(f"Error {error_count}/5")
                    frame = last_valid
                    
                    if error_count >= 5:
                        print("Reinitializing...")
                        break
                else:
                    error_count = 0
                    last_valid = frame.copy()
                
                cv2.imshow('CAM1', frame)
                if cv2.waitKey(1) == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                
            cap.release()
        else:
            print("Open failed, retrying...")
            time.sleep(2)

if __name__ == "__main__":
    main()
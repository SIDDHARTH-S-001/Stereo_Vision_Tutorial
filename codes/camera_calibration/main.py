import cv2
import numpy as np
from stereo_vision import StereoVision

def load_camera_parameters(cam_idx):
    """Load camera matrix and distortion coefficients."""
    # Load from files (replace with your actual file paths)
    camera_matrix = np.load(f'camera_matrix_{cam_idx}.txt')
    distortion_coeff = np.load(f'distortion_coefficients_{cam_idx}.txt')
    return {'matrix': camera_matrix, 'distortion': distortion_coeff}

def load_stereo_parameters():
    """Load stereo calibration parameters."""
    # Replace these with your actual stereo calibration results
    # These are example values - you need to calibrate your stereo setup
    R = np.eye(3)  # Rotation matrix between cameras
    T = np.array([[-0.1], [0.0], [0.0]])  # Translation vector (baseline)
    E = np.eye(3)  # Essential matrix
    F = np.eye(3)  # Fundamental matrix
    
    return {'R': R, 'T': T, 'E': E, 'F': F}

def main():
    # Load camera parameters
    cam0_params = load_camera_parameters(0)
    cam1_params = load_camera_parameters(1)
    
    # Load stereo parameters
    stereo_params = load_stereo_parameters()
    
    # Initialize stereo vision system
    stereo = StereoVision(cam0_params, cam1_params, stereo_params)
    
    # Open cameras
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)
    
    # Set camera properties (adjust based on your cameras)
    for cap in [cap0, cap1]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    while True:
        # Read frames
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if not ret0 or not ret1:
            print("Error reading frames")
            break
        
        # Process frames
        results = stereo.process_frames(frame0, frame1)
        
        # Normalize disparity and depth for display
        disp_norm = cv2.normalize(results['disparity'], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_norm = cv2.normalize(results['depth'], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        
        # Display results
        cv2.imshow('Camera 0', results['frame0'])
        cv2.imshow('Camera 1', results['frame1'])
        cv2.imshow('Rectified 0', results['frame0_rect'])
        cv2.imshow('Rectified 1', results['frame1_rect'])
        cv2.imshow('Disparity', disp_norm)
        cv2.imshow('Depth', depth_colormap)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
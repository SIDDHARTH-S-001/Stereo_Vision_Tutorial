import cv2
import numpy as np

class StereoVision:
    def __init__(self, cam0_id=0, cam1_id=1):
        self.cam0 = cv2.VideoCapture(cam0_id, cv2.CAP_ANY)
        self.cam1 = cv2.VideoCapture(cam1_id, cv2.CAP_ANY)
        
        if not self.cam0.isOpened() or not self.cam1.isOpened():
            raise Exception("Could not open video devices")
            
        # Load calibration data
        self.camera_matrix_0 = np.loadtxt('camera_matrix_0.txt')
        self.dist_coeffs_0 = np.loadtxt('distortion_coefficients_0.txt')
        self.camera_matrix_1 = np.loadtxt('camera_matrix_1.txt')
        self.dist_coeffs_1 = np.loadtxt('distortion_coefficients_1.txt')
        
        # Stereo rectification parameters (you may need to load or calculate these)
        self.R = np.eye(3)  # Replace with your rotation matrix if available
        self.T = np.zeros(3)  # Replace with your translation vector if available
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.camera_matrix_0, self.dist_coeffs_0,
            self.camera_matrix_1, self.dist_coeffs_1,
            (640, 480), self.R, self.T)
        
        # Initialize stereo block matching
        self.stereo = cv2.StereoBM_create()
        self.stereo.setNumDisparities(16)
        self.stereo.setBlockSize(15)
        
    def undistort_rectify(self, frame, cam_id):
        if cam_id == 0:
            map1, map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix_0, self.dist_coeffs_0, self.R1,
                self.P1, (frame.shape[1], frame.shape[0]), cv2.CV_32FC1)
        else:
            map1, map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix_1, self.dist_coeffs_1, self.R2,
                self.P2, (frame.shape[1], frame.shape[0]), cv2.CV_32FC1)
        
        return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    
    def compute_disparity(self, frame0, frame1):
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        disparity = self.stereo.compute(gray0, gray1)
        return disparity
    
    def compute_depth(self, disparity):
        depth = cv2.reprojectImageTo3D(disparity, self.Q)
        depth_map = cv2.normalize(depth[:,:,2], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return depth_map
    
    def process_frames(self):
        while True:
            ret0, frame0 = self.cam0.read()
            ret1, frame1 = self.cam1.read()
            
            if not ret0 or not ret1:
                print("Failed to grab frames")
                break
                
            # Undistort and rectify frames
            frame0_rect = self.undistort_rectify(frame0, 0)
            frame1_rect = self.undistort_rectify(frame1, 1)
            
            # Compute disparity
            disparity = self.compute_disparity(frame0_rect, frame1_rect)
            
            # Compute depth
            depth_map = self.compute_depth(disparity)
            
            # Display results
            cv2.imshow('Camera 0', frame0_rect)
            cv2.imshow('Camera 1', frame1_rect)
            cv2.imshow('Disparity', cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
            cv2.imshow('Depth Map', depth_map)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    def release(self):
        self.cam0.release()
        self.cam1.release()
        cv2.destroyAllWindows()
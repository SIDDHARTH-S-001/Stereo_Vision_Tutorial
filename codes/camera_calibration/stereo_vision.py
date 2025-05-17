import cv2
import numpy as np

class StereoVision:
    def __init__(self, cam0_params, cam1_params, stereo_params):
        """
        Initialize stereo vision system with camera and stereo parameters.
        
        Args:
            cam0_params: Dict containing 'matrix' and 'distortion' for camera 0
            cam1_params: Dict containing 'matrix' and 'distortion' for camera 1
            stereo_params: Dict containing stereo calibration parameters
        """
        # Camera parameters
        self.cam0_matrix = cam0_params['matrix']
        self.cam0_dist = cam0_params['distortion']
        self.cam1_matrix = cam1_params['matrix']
        self.cam1_dist = cam1_params['distortion']
        
        # Stereo parameters
        self.R = stereo_params['R']
        self.T = stereo_params['T']
        self.E = stereo_params['E']
        self.F = stereo_params['F']
        
        # Initialize stereo rectification maps
        self.init_rectification_maps()
        
        # Initialize stereo block matcher
        self.init_block_matcher()
    
    def init_rectification_maps(self):
        """Compute rectification transforms and projection matrices."""
        img_size = (640, 480)  # Adjust based on your camera resolution
        flags = cv2.CALIB_ZERO_DISPARITY
        alpha = 0  # 0=cropped, 1=uncropped
        
        # Stereo rectify
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.cam0_matrix, self.cam0_dist,
            self.cam1_matrix, self.cam1_dist,
            img_size, self.R, self.T,
            alpha=alpha, flags=flags
        )
        
        # Compute rectification maps
        self.map0x, self.map0y = cv2.initUndistortRectifyMap(
            self.cam0_matrix, self.cam0_dist, self.R1, self.P1, img_size, cv2.CV_32FC1
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.cam1_matrix, self.cam1_dist, self.R2, self.P2, img_size, cv2.CV_32FC1
        )
    
    def init_block_matcher(self):
        """Initialize the stereo block matcher."""
        # Parameters for StereoBM
        self.block_matcher = cv2.StereoBM_create()
        self.block_matcher.setNumDisparities(16 * 5)  # Must be divisible by 16
        self.block_matcher.setBlockSize(21)  # Odd number between 5-255
        
        # Parameters for StereoSGBM (alternative)
        # self.block_matcher = cv2.StereoSGBM_create(
        #     minDisparity=0,
        #     numDisparities=16 * 5,
        #     blockSize=5,
        #     P1=8 * 3 * 5**2,
        #     P2=32 * 3 * 5**2,
        #     disp12MaxDiff=1,
        #     uniquenessRatio=10,
        #     speckleWindowSize=100,
        #     speckleRange=32,
        #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        # )
    
    def undistort_rectify(self, frame0, frame1):
        """Undistort and rectify stereo frames."""
        # Undistort and rectify frames
        frame0_rect = cv2.remap(frame0, self.map0x, self.map0y, cv2.INTER_LINEAR)
        frame1_rect = cv2.remap(frame1, self.map1x, self.map1y, cv2.INTER_LINEAR)
        
        return frame0_rect, frame1_rect
    
    def compute_disparity(self, frame0, frame1):
        """Compute disparity map from stereo frames."""
        # Convert to grayscale if needed
        if len(frame0.shape) == 3:
            frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        if len(frame1.shape) == 3:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity
        disparity = self.block_matcher.compute(frame0, frame1)
        disparity = disparity.astype(np.float32) / 16.0  # Convert to actual disparity values
        
        return disparity
    
    def compute_depth(self, disparity):
        """Compute depth map from disparity using Q matrix."""
        # Reproject to 3D (depth is the Z coordinate)
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        depth = points_3d[:, :, 2]
        
        # Filter invalid points (set to 0)
        depth[disparity <= 0] = 0
        
        return depth
    
    def process_frames(self, frame0, frame1):
        """Process stereo frames and return all results."""
        # Undistort and rectify
        frame0_rect, frame1_rect = self.undistort_rectify(frame0, frame1)
        
        # Compute disparity
        disparity = self.compute_disparity(frame0_rect, frame1_rect)
        
        # Compute depth
        depth = self.compute_depth(disparity)
        
        return {
            'frame0': frame0,
            'frame1': frame1,
            'frame0_rect': frame0_rect,
            'frame1_rect': frame1_rect,
            'disparity': disparity,
            'depth': depth
        }
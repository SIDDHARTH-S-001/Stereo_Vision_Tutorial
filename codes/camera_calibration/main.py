from stereo_vision import StereoVision

def main():
    stereo_vision = StereoVision(cam0_id=0, cam1_id=1)
    try:
        stereo_vision.process_frames()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        stereo_vision.release()

if __name__ == "__main__":
    main()
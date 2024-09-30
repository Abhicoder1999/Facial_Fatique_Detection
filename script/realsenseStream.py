import pyrealsense2 as rs
import numpy as np
import cv2
from time import sleep


class Camera:
    def __init__(self) -> None:
        self.pipeline = rs.pipeline()
        # Create a configuration object to configure the pipeline
        config = rs.config()
        # Configure the pipeline to stream RGB video
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # Start streaming
        self.pipeline.start(config)

        # Retrieve camera intrinsics
        profile = self.pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color)  # Fetch the stream profile for the color stream
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()  # Get the intrinsics

    def get_camera_intrinsics(self):
        return {
            "width": self.intrinsics.width,
            "height": self.intrinsics.height,
            "ppx": self.intrinsics.ppx,  # Principal point x
            "ppy": self.intrinsics.ppy,  # Principal point y
            "fx": self.intrinsics.fx,    # Focal length x
            "fy": self.intrinsics.fy,    # Focal length y
            "distortion_model": self.intrinsics.model,
            "coeffs": self.intrinsics.coeffs,  # Distortion coefficients
        }


    def run(self,view_img):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert the RealSense frame to a NumPy array
        color_image = np.asanyarray(color_frame.get_data())


        if view_img == True:
            cv2.imshow("RealSense D435 RGB", color_image)
            cv2.waitKey(1)

        return color_image
       

    def __del__(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cm = Camera()
    print(cm.get_camera_intrinsics())
    while True:
        img = cm.run(True)
        
            

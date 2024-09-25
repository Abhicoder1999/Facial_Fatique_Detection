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

    while True:
        img = cm.run(True)
        
            

import numpy as np
import pyrealsense2 as rs

use_bag = False

# Declare RealSense pipeline, encapsulating the actual device and sensors
pipe = rs.pipeline()

# Build config object and request pose data
cfg = rs.config()

if use_bag:
    cfg.enable_device_from_file('data_files/capture_1_t265.bag')
else:
    cfg.enable_stream(rs.stream.pose)
    cfg.enable_record_to_file('data_files/capture_1_t265.bag')

# Start streaming with requested config
pipe.start(cfg)

def get_pose():
    frames = pipe.wait_for_frames()
    pose = frames.get_pose_frame()
    if pose:
        data = pose.get_pose_data()
        return np.array([data.translation.x, data.translation.y, data.translation.z]), np.array([data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w])
    
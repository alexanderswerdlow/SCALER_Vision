import os
import sys
import numpy as np
import pyrealsense2 as rs
import math as m
from scipy.spatial.transform import Rotation as R

# Declare RealSense pipeline, encapsulating the actual device and sensors
pipe = rs.pipeline()

# Build config object and request pose data
cfg = rs.config()
cfg.enable_stream(rs.stream.pose)

# Start streaming with requested config
pipe.start(cfg)

def get_world_camera_tf():
    frames = pipe.wait_for_frames()
    pose = frames.get_pose_frame()
    if pose:
        data = pose.get_pose_data()

        # TODO: Maybe invert as below?
        # w = data.rotation.w
        # x = -data.rotation.z
        # y = data.rotation.x
        # z = -data.rotation.y

        # R_ = R.from_quat([data.rotation.w,
		# 								 data.rotation.x,
		# 								 data.rotation.y,
		# 								 data.rotation.z]).as_matrix()
        # t = data.translation
        # t = np.array([t.x,t.y,t.z]).reshape((1,3))
        # T = np.zeros((4,4))
        # T[:3,:3] = R_
        # T[:3,3] = t
        # T[3,3] = 1.0

        R_ = R.from_quat([data.rotation.w,
										 -data.rotation.z,
										 data.rotation.x,
										 -data.rotation.y]).as_matrix()
        t = data.translation
        t = np.array([t.x,t.y,t.z]).reshape((1,3))
        T = np.zeros((4,4))
        T[:3,:3] = R_
        T[:3,3] = t
        T[3,3] = 1.0

        # return np.array([data.translation.x, -data.translation.y, -data.translation.z]), np.array([x, y, z, w])
        return T
        # geometry.rigid_transform(vole,np.linalg.inv(transforms[i]))


def get_pose():
    frames = pipe.wait_for_frames()
    pose = frames.get_pose_frame()
    if pose:
        data = pose.get_pose_data()
        return np.array([data.translation.x, data.translation.y, data.translation.z]), np.array([data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w])
    
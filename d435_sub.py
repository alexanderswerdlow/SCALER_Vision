import numpy as np
import pyrealsense2 as rs
import cv2
from util import IncrementalNpzWriter

pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == "RGB Camera":
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)

# [ 1280x720  p[655.67 358.885]  f[906.667 906.783]

profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

pc = rs.pointcloud()
colorizer = rs.colorizer()

align_to = rs.stream.color
align = rs.align(align_to)

def get_rgbd():
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    ir_frame = frames.get_infrared_frame()

    # Grab new intrinsics (may be changed by decimation)
    depth_intrinsics = rs.video_stream_profile(
        depth_frame.profile).get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    ir_image = np.asanyarray(ir_frame.get_data())

    # points = pc.calculate(depth_frame)
    # pc.map_to(color_frame)

    # v, t = points.get_vertices(), points.get_texture_coordinates()
    # verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
    # texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(verts)
    # o3d.visualization.draw_geometries([pcd])


    return (color_image, depth_image, ir_image)

def view():
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    ir_colormap = cv2.applyColorMap(cv2.convertScaleAbs(ir_image, alpha=0.03), cv2.COLORMAP_JET)
    images = np.hstack((color_image, depth_colormap, ir_colormap))

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    cv2.waitKey(1)

if __name__ == "__main__":
    #writer = IncrementalNpzWriter("rgbd.npz")
    idx = 0
    try:
        while True:
            print(f"Loop {idx}")
            color_image, depth_image, ir_image = get_rgbd()
            if color_image is None:
                continue
            view()
            # color_image = np.dstack((color_image, depth_image))
            # writer.write(f"{idx}", color_image)
            idx += 1

    finally:
        #writer.close()
        pipeline.stop()

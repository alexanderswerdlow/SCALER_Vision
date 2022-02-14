import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_transformation(trans, rot):
    r = R.from_quat(rot)
    rotation = np.array(r.as_matrix())
    translation = trans[np.newaxis].T
    T = np.hstack((rotation, translation))
    T = np.vstack((T, np.array([0, 0, 0, 1])))
    return T

T265_to_D435_mat = np.array([[0.999968402, -0.006753626, -0.004188075, -0.015890727],
                      [-0.006685408, -0.999848172, 0.016093893, 0.028273059],
                      [-0.004296131, -0.016065384, -0.999861654, -0.009375589],
                      [0, 0, 0, 1]])

loaded = np.load(f"data_files/rgbd1.npz", allow_pickle=True)
frames = list(loaded.keys())
transformation_matrix_set = []
d435_data_list_all = []
for frame_key in frames:
    color_image, depth_image, ir_image, trans, rot = loaded[frame_key]
    transformation_matrix_set.append(get_transformation(trans, rot) @ T265_to_D435_mat)
    d435_data_list_all.append((color_image, depth_image))


def get_geom_pcl(rgb_im, depth_im):
    im_rgb = o3d.geometry.Image(np.ascontiguousarray(rgb_im))
    im_depth = o3d.geometry.Image(np.ascontiguousarray(depth_im))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(im_rgb, im_depth, convert_rgb_to_intensity=False)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(1280, 720, 906.667, 906.783, 655.67, 358.885)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, project_valid_depth_only=True)
    return pcd

pcds = []
for i in range(len(frames)):
    pcl = get_geom_pcl(*d435_data_list_all[i])
    T = transformation_matrix_set[i]
    pcl.transform(T)
    pcds.append(pcl)

o3d.visualization.draw_geometries(pcds)

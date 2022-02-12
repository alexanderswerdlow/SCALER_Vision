import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.rotations import active_matrix_from_intrinsic_euler_xyz
from pytransform3d.transformations import transform_from, plot_transform
from pytransform3d.camera import make_world_grid, world2image, plot_camera, make_world_line
from pytransform3d import transformations as pt


loaded = np.load(f"data_files/rgbd.npz", allow_pickle=True)
frames = list(loaded.keys())
_, _, _, trans, rot = loaded["0"]

initcam2world = pt.transform_from_pq(np.hstack((np.array([trans[0], -trans[1], -trans[2]]), np.array([rot[3], rot[0], rot[2], rot[1]]))))
_, _, _, trans, rot = loaded["12"]

rot_fixed = np.array([rot[3], rot[0], rot[2], rot[1]]) # stock
rot_fixed = np.array([rot[3], rot[2], rot[0], rot[1]]) # rpy docs
rot_fixed = np.array([rot[3], rot[0], rot[2], rot[1]]) # rpy docs
rot_fixed = np.array([rot[3], -rot[2], rot[0], -rot[1]])

cam2world = pt.transform_from_pq(np.hstack((np.array([trans[0], -trans[1], -trans[2]]), rot_fixed)))
from scipy.spatial.transform import Rotation as R


#print(R.from_quat(rot).as_euler('xyz', degrees=True))
#print(R.from_quat(np.array([rot[0], rot[1], rot[3], rot[2]])).as_euler('xyz', degrees=True))

x,y,z,w = rot[0], rot[1], rot[2], rot[3]

import math as m
pitch =  -m.asin(2.0 * (x*z - w*y)) * 180.0 / m.pi;
roll  =  m.atan2(2.0 * (w*x + y*z), w*w - x*x - y*y + z*z) * 180.0 / m.pi;
yaw   =  m.atan2(2.0 * (w*z + x*y), w*w + x*x - y*y - z*z) * 180.0 / m.pi;

# print("RPY [deg]: Roll: {0:.7f}, Pitch: {1:.7f}, Yaw: {2:.7f}".format(roll, pitch, yaw))


focal_length = 906.667
sensor_size = np.array([1280, 720])
image_size = np.array([1280, 720])
intrinsic_camera_matrix = np.array([[906.667, 0, 655.67], [0, 906.783, 358.885], [0, 0, 1]])
n_lines = 20
n_points_per_line = 50
xlim = (-0.25, 0.25)
ylim = (-0.25, 0.25)
world_grid_x = np.vstack([make_world_line([xlim[0], y], [xlim[1], y], n_points_per_line) for y in np.linspace(ylim[0], ylim[1], n_lines)])
world_grid_y = np.vstack([make_world_line([x, ylim[0]], [x, ylim[1]], n_points_per_line) for x in np.linspace(xlim[0], xlim[1], n_lines)])
world_grid = np.vstack((world_grid_x, world_grid_y))
world_grid[:, 2] = 0.50

num_cloud = 40
world_grid = np.empty((num_cloud, 4))
world_grid[:,] = np.array([0.0623821247, 0.1240413574, 0.5030982574, 1])
world_grid_ = np.random.uniform(-0.02, 0.02, (num_cloud, 3))
X0 = np.ones((num_cloud,1))
world_grid = world_grid + np.hstack((world_grid_,X0))

image_grid = world2image(world_grid, cam2world, sensor_size, image_size, focal_length, kappa=0.4)
image_grid_init = world2image(world_grid, initcam2world, sensor_size, image_size, focal_length, kappa=0.4)

plt.figure(figsize=(12, 5))
ax = make_3d_axis(1, 131, unit="m")
ax.view_init(elev=30, azim=-70)
plot_transform(ax)
plot_transform(ax, A2B=cam2world, s=0.3, name="Camera (t=18)")
plot_camera(ax, intrinsic_camera_matrix, cam2world, sensor_size=sensor_size, virtual_image_distance=0.5)

plot_transform(ax, A2B=initcam2world, s=0.3, name="Camera (t=0)")
plot_camera(ax, intrinsic_camera_matrix, initcam2world, sensor_size=sensor_size, virtual_image_distance=0.5)

ax.set_title("Camera and world frames")
ax.scatter(world_grid[:, 0], world_grid[:, 1], world_grid[:, 2], s=1, alpha=0.2)
ax.scatter(0, 0, world_grid[-1, 2], color="r")
ax.view_init(elev=25, azim=-130)

ax = plt.subplot(132, aspect="equal")
ax.set_title("Init Camera image (t=0)")
ax.set_xlim(0, image_size[0])
ax.set_ylim(0, image_size[1])
ax.scatter(image_grid_init[:, 0], -(image_grid_init[:, 1] - image_size[1]))
ax.scatter(image_grid_init[-1, 0], -(image_grid_init[-1, 1] - image_size[1]), color="r")

ax = plt.subplot(133, aspect="equal")
ax.set_title("Camera image (t=18)")
ax.set_xlim(0, image_size[0])
ax.set_ylim(0, image_size[1])
ax.scatter(image_grid[:, 0], -(image_grid[:, 1] - image_size[1]))
ax.scatter(image_grid[-1, 0], -(image_grid[-1, 1] - image_size[1]), color="r")


plt.show()

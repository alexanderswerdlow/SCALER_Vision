import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.rotations import active_matrix_from_intrinsic_euler_xyz, quaternion_wxyz_from_xyzw
from pytransform3d.transformations import transform_from, plot_transform
from pytransform3d.camera import make_world_grid, world2image, plot_camera, make_world_line
from pytransform3d import transformations as pt
from scipy.spatial.transform import Rotation as R

loaded = np.load(f"data_files/rgbd1.npz", allow_pickle=True)
frames = list(loaded.keys())
_, _, _, trans, rot = loaded["0"]
initcam2world = pt.transform_from_pq(np.hstack((np.array([trans[0], -trans[1], -trans[2]]), np.array([rot[3], rot[0], rot[2], rot[1]]))))

def get_transformation(trans, rot):
    r = R.from_quat(rot)
    rotation = np.array(r.as_matrix())
    translation = trans[np.newaxis].T
    T = np.hstack((rotation, translation))
    T = np.vstack((T, np.array([0, 0, 0, 1])))
    return T

T_d_wrt_t = np.array([[0.999968402, -0.006753626, -0.004188075, -0.015890727],
                        [-0.006685408, -0.999848172, 0.016093893, 0.028273059],
                        [-0.004296131, -0.016065384, -0.999861654, -0.009375589],
                        [0, 0, 0, 1]])



for frame_key in frames:
    color_image, _, _, trans, rot = loaded[frame_key]

    options = [
        (np.array([trans[0], -trans[1], -trans[2]]), np.array([rot[0], rot[2], rot[1], rot[3]])),  # stock
        # (np.array([trans[0], -trans[1], -trans[2]]), np.array([rot[2], rot[0], rot[1], rot[3]])),  # rpy docs
        # (np.array([trans[0], -trans[1], -trans[2]]), np.array([rot[0], rot[2], rot[1], rot[3]])),
        # (np.array([trans[0], -trans[1], -trans[2]]), np.array([-rot[2], rot[0], -rot[1], rot[3]])),
    ]


    # loaded = np.load(f"data_files/rgbd1.npz", allow_pickle=True)
    # frames = list(loaded.keys())
    # transformation_matrix_set = []
    # d435_data_list_all = []
    # for frame_key in frames:
    #     color_image, depth_image, ir_image, trans, rot = loaded[frame_key]
    #     transformation_matrix_set.append(get_transformation(trans, rot) @ T_d_wrt_t)
    #     d435_data_list_all.append((color_image, depth_image))


    for idx, (trans, rot) in enumerate(options):
        # cam2world = pt.transform_from_pq(np.hstack((trans, quaternion_wxyz_from_xyzw(rot))))
        cam2world = np.linalg.inv(get_transformation(trans, rot) @ T_d_wrt_t)

        focal_length = 906.667
        sensor_size = np.array([1280, 720])
        image_size = np.array([1280, 720])
        intrinsic_camera_matrix = np.array([[906.667, 0, 655.67], [0, 906.783, 358.885], [0, 0, 1]])

        # Create plane at z = 0.5
        # n_lines = 20
        # n_points_per_line = 50
        # xlim = (-0.25, 0.25)
        # ylim = (-0.25, 0.25)
        # world_grid_x = np.vstack([make_world_line([xlim[0], y], [xlim[1], y], n_points_per_line) for y in np.linspace(ylim[0], ylim[1], n_lines)])
        # world_grid_y = np.vstack([make_world_line([x, ylim[0]], [x, ylim[1]], n_points_per_line) for x in np.linspace(xlim[0], xlim[1], n_lines)])
        # world_grid = np.vstack((world_grid_x, world_grid_y))
        # world_grid[:, 2] = 0.50

        # Create cloud around estimated centroid
        num_cloud = 40
        world_grid = np.empty((num_cloud, 4))
        world_grid[:,] = np.array([0.0623821247, 0.1240413574, 0.5030982574, 1])
        world_grid += np.hstack((np.random.uniform(-0.02, 0.02, (num_cloud, 3)), np.ones((num_cloud, 1))))

        plt.figure(figsize=(12, 5))
        ax = make_3d_axis(1, 131, unit="m")

        plot_transform(ax)
        plot_transform(ax, A2B=cam2world, s=0.3, name=f"Camera (t={frame_key}")
        plot_camera(ax, intrinsic_camera_matrix, cam2world, sensor_size=sensor_size, virtual_image_distance=0.5)

        # plot_transform(ax, A2B=initcam2world, s=0.3, name="Camera (t=0)")
        # plot_camera(ax, intrinsic_camera_matrix, initcam2world, sensor_size=sensor_size, virtual_image_distance=0.5)

        ax.set_title("Camera and world frames")
        ax.scatter(world_grid[:, 0], world_grid[:, 1], world_grid[:, 2], s=1, alpha=0.2)
        ax.scatter(0, 0, world_grid[-1, 2], color="r")
        ax.view_init(elev=270, azim=270)
        ax.set_xlim3d([-0.5, 0.5])
        ax.set_ylim3d([-0.5, 0.5])

        # image_grid_init = world2image(world_grid, initcam2world, sensor_size, image_size, focal_length, kappa=0.4)
        # ax = plt.subplot(142, aspect="equal")
        # ax.set_title("Init Camera image (t=0)")
        # ax.set_xlim(0, image_size[0])
        # ax.set_ylim(0, image_size[1])
        # ax.scatter(image_grid_init[:, 0], -(image_grid_init[:, 1] - image_size[1]))
        # ax.scatter(image_grid_init[-1, 0], -(image_grid_init[-1, 1] - image_size[1]), color="r")

        image_grid = world2image(world_grid, cam2world, sensor_size, image_size, focal_length, kappa=0.4)
        ax = plt.subplot(132, aspect="equal")
        ax.set_title(f"Camera image (t={frame_key})")
        ax.set_xlim(0, image_size[0])
        ax.set_ylim(0, image_size[1])
        ax.scatter(image_grid[:, 0], -(image_grid[:, 1] - image_size[1]))
        ax.scatter(image_grid[-1, 0], -(image_grid[-1, 1] - image_size[1]), color="r")

        ax = plt.subplot(133, aspect="equal")
        ax.set_title(f"Camera image real (t={frame_key})")
        ax.imshow(color_image)

        plt.savefig(f"output/{frame_key}-transform-{idx}.png")
        plt.close()




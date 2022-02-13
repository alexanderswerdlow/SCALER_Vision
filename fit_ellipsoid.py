import enum
import os
import cv2
import time
import numpy as np
import outer_ellipsoid
import matplotlib.pyplot as plt
import open3d as o3d
import pickle
import argparse
import numpy.linalg as la
from scipy.spatial.transform import Rotation as R


def rgbd_to_pcl(rgb_im, depth_im, param, vis=False):
    """
    Calibration:  [ 1280x720  p[655.67 358.885]  f[906.667 906.783]  Inverse Brown Conrady [0 0 0 0 0] ] [0.0, 0.0, 0.0, 0.0, 0.0]
    """
    im_rgb = o3d.geometry.Image(np.ascontiguousarray(rgb_im))
    im_depth = o3d.geometry.Image(np.ascontiguousarray(depth_im))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(im_rgb, im_depth, convert_rgb_to_intensity=False)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(1280, 720, 906.667, 906.783, 655.67, 358.885)
    _, extrinsic = param
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, project_valid_depth_only=True)
#     pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    if vis:
        o3d.visualization.draw_geometries([pcd])
    return pcd


def segment_image(im, save_detections):
    """Run Mask-RCNN w/Detectron 2 and return bounding boxes, masks, scores"""
    start = time.time()
    outputs = predictor(im)

    if len(outputs["instances"].pred_boxes) == 0:
        print(f"No detections for image {frame_key}")
        cv2.imwrite(f"output/{frame_key}-climbnet.jpg", im)
        return None

    results = outputs["instances"].to("cpu")
    boxes = results.pred_boxes.tensor.detach().numpy()
    scores = results.scores.detach().numpy()
    masks = results.pred_masks.detach().numpy()

    if save_detections:
        print(f"Segmentation took {time.time() - start}")
        v = Visualizer(
            im[:, :, ::-1],
            metadata=train_metadata,
            scale=0.75,
            instance_mode=ColorMode.IMAGE,
        )
        cv2.imwrite(f"output/{frame_key}-climbnet.jpg", v.draw_instance_predictions(results).get_image()[:, :, ::-1])

    return scores, boxes, masks


def cluster_and_fit(im, depth, param, scores, boxes, masks, save_detections):
    """Create point cloud based on segmented masks, cluster, and fit ellipsoids"""
    # Empty images to put detected regions
    detected_rgb, detected_depth = np.zeros_like(im), -np.ones_like(depth)
    for idx, mask in enumerate(masks):
        if scores[idx] > 0.98:
            detected_rgb[mask] = im[mask]
            detected_depth[mask] = depth[mask]

    # cv2.imshow('a', detected_rgb)
    # cv2.waitKey(0)
    # cv2.imshow('ba', depth)
    # cv2.waitKey(0)

    # Generate point cloud from RGBD
    pcd = rgbd_to_pcl(detected_rgb, detected_depth, param, vis=False)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if len(pcd.points) == 0:
        return None

    # Segment point cloud, returning (n,) array of labels
    start = time.time()
    labels = np.array(pcd.cluster_dbscan(eps=0.004, min_points=10, print_progress=False))

    if save_detections:
        print(f"Clustering points for frame {frame_key} took {time.time() - start}")
        background_rgb, background_depth = im.copy(), depth.copy()  # Background images with detected regions removed
        for idx, mask in enumerate(masks):
            if scores[idx] > 0.98:
                background_rgb[mask] = [0, 0, 0]
                background_depth[mask] = -1

        fig = plt.figure(figsize=(8.0, 8.0))
        ax = fig.add_subplot(111, projection="3d")

        # pcd_background = rgbd_to_pcl(background_rgb, background_depth, vis=False).voxel_down_sample(voxel_size=0.01)
        # points = np.asarray(pcd_background.points)
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.asarray(pcd_background.colors), label=f"Background", alpha=0.1, s=0.5)

    if args.vis:
        vis.clear_geometries()

    ellipsoids = []
    for idx in range(labels.max() + 1):
        cluster_pcd = pcd.select_by_index(np.argwhere(labels == idx))
        if args.vis:
            vis.add_geometry(cluster_pcd)
        
        if cluster_pcd.get_axis_aligned_bounding_box().get_extent().max() < 0.1:
            continue
        
        cluster_pcd = cluster_pcd.voxel_down_sample(voxel_size=0.005)
        points = np.asarray(cluster_pcd.points)
        

        try:
            start = time.time()
            A_outer, centroid_outer = outer_ellipsoid.outer_ellipsoid_fit(points, tol=0.01)
            ellipsoids.append((A_outer, centroid_outer))
        except:
            print(f"Fit failed for {idx}")
            continue

        if save_detections:
            print(f"Fit for {idx} took {time.time() - start} for {len(points)} points")
            outer_ellipsoid.plot_ellipsoid(A_outer, centroid_outer, "green", ax)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.asarray(cluster_pcd.colors), label=f"{idx}")

    if save_detections:
        ax.view_init(elev=270, azim=270)
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        plt.legend(loc="best")
        plt.savefig(f"output/{frame_key}.jpg", dpi=300, bbox_inches="tight")
        plt.close()
    
    if args.vis:
        vis.poll_events()
        vis.update_renderer()

    return ellipsoids


def run_pipeline(color_image, depth_image, param, detection=None):
    start = time.time()
    if detection:
        scores, boxes, masks = detection
    else:
        if (detection := segment_image(color_image, False)) is None:
            return None, None
        scores, boxes, masks = detection

    ellipsoids = cluster_and_fit(color_image, depth_image, param, scores, boxes, masks, False)
    # print(f"Frame {frame_key} took {time.time() - start}")
    return ellipsoids, detection


def create_transform_matrix(rotation, translation):
    '''return a 4 x 4 transform matrix'''
    return np.block([
                    [rotation, translation[:, np.newaxis]],
                    [np.zeros((1,3)), 1]
                    ])

def rigid_transform(points,T):
    pnum = points.shape[0]
    homo_points = np.concatenate([points,np.ones((pnum,1))],axis=1)
    return (T @ homo_points.T).T[:,:3]

if __name__ == "__main__":
    verbose = False
    parser = argparse.ArgumentParser(description="Run Ellipsoid Fitting")
    parser.add_argument("--load_segmentation", dest="load_segmentation", action="store_true")
    parser.add_argument("--save_segmentation", dest="save_segmentation", action="store_true")
    parser.add_argument("--load_rgbd", type=str, default="rgbd.npz")
    parser.add_argument("--use_t265", dest="use_t265", action="store_true")
    parser.add_argument("--use_d435", dest="use_d435", action="store_true")
    parser.add_argument("--run_from_file", dest="run_from_file", action="store_true")
    parser.add_argument("--vis", dest="vis", action="store_true")
    parser.add_argument("--data_files", type=str, default="data_files")
    args = parser.parse_args()

    if args.load_segmentation or args.run_from_file:
        try:
            detections = pickle.load(open(f"{args.data_files}/detections.p", "rb"))
        except (OSError, IOError) as e:
            print("detections.p not found, exiting")
            exit()
    else:
        print("Setting up detectron...")
        import detectron2 as dt2
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.utils.visualizer import ColorMode, Visualizer
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor

        model_path = f"{args.data_files}/model_d2_R_50_FPN_3x.pth"
        dt2.data.datasets.register_coco_instances("climb_dataset", {}, f"{args.data_files}/mask.json", "")

        cfg = dt2.config.get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATALOADER.NUM_WORKERS = 1
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (hold, volume, downclimb)
        cfg.MODEL.WEIGHTS = os.path.join(model_path)
        # cfg.MODEL.DEVICE = "cpu"
        cfg.DATASETS.TEST = ("climb_dataset",)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
        predictor = DefaultPredictor(cfg)
        train_metadata = MetadataCatalog.get("climb_dataset")
        DatasetCatalog.get("climb_dataset")
        detections = {}

    if args.vis:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    if args.run_from_file:
        loaded = np.load(f'{args.data_files}/{args.load_rgbd}', allow_pickle=True)
        frames = list(loaded.keys())

        T265_to_D435_trans = np.array([0.009, 0.021, 0.027]) #translation in meters
        T265_to_D435_rot = np.array([0.000, -0.018, 0.005]) #rpy in radians
        #XYZ represents intrinic rotation which is roll, pitch and yaw
        T265_to_D435_rot = R.from_euler('xyz', T265_to_D435_rot).as_matrix()
        T265_to_D435_mat = create_transform_matrix(T265_to_D435_rot, T265_to_D435_trans)

        for frame_key in frames:
            color_image, depth_image, ir_image, trans, rot = loaded[frame_key]
            print(frame_key)
            # print(trans, rot)
            from PIL import Image
            im = Image.fromarray(color_image)
            im.save(f'output/{frame_key}.png')
            rot = np.array([-rot[2],rot[0], -rot[1],rot[3]])
            rot = R.from_quat(rot).as_matrix()
            trans = np.array([trans[0], -trans[1], -trans[2]])
            extrinsic = create_transform_matrix(rot, trans)
            ellipsoids, detection = run_pipeline(color_image, depth_image, (None, extrinsic), detections[frame_key])
            if ellipsoids is None:
                continue

            for A, centroid in ellipsoids:
                _, D, V = la.svd(A)
                rx, ry, rz = 1.0 / np.sqrt(D)
                print(centroid * 100)

        exit()

    if args.use_d435:
        import d435_sub

        if args.use_t265:
            import t265_sub

        from pytransform3d import rotations as pr
        from pytransform3d import transformations as pt
        from pytransform3d.transform_manager import TransformManager

        tm = TransformManager()
        

        # Change t265 localization to d435 frame
        T265_to_D435 = np.array([
        [0.999968402, -0.006753626, -0.004188075, -0.015890727],
        [-0.006685408, -0.999848172, 0.016093893, 0.028273059],
        [-0.004296131, -0.016065384, -0.999861654, -0.009375589],
        [0.0,0.0,0.0,1.0]])


        # H_t265_d400 = np.array([
        #     [1, 0, 0, 0],
        #     [0, -1.0, 0, 0],
        #     [0, 0, -1.0, 0],
        #     [0, 0, 0, 1]])
        # T265_to_D435 = H_t265_d400 @ T265_to_D435_mat
        # print(T265_to_D435_mat)


    
        frame_key = 0
        all_ellipsoids = []
        while True:
            if frame_key < 10:
                frame_key += 1
                continue
            #input("Press any key to take picture")
            (color_image, depth_image, _), (intrinsic, _) = d435_sub.get_rgbd()


            # extrinsic = np.linalg.inv(np.linalg.inv(H_t265_d400) @ extrinsic @ H_t265_d400))
            # extrinsic = R_Standard_d400 @ np.linalg.inv(extrinsic)
            # print(extrinsic)

            # T = t265_sub.get_world_camera_tf()
            T265_tran, T265_quat = t265_sub.get_pose()
            T = create_transform_matrix(rotation=R.from_quat(T265_quat).as_matrix(), translation=T265_tran)

            world_to_T265_init = pt.transform_from(pr.active_matrix_from_intrinsic_euler_xyz(np.array([np.pi/2, 0.0, 0])),np.zeros(3))

            tm.add_transform("world", "T265_init", world_to_T265_init)
            tm.add_transform("T265_init", "T265", T)
            tm.add_transform("T265", "D435", T265_to_D435)
           
            
            #continue

            param = (intrinsic, T)
            ellipsoids, detection = run_pipeline(color_image, depth_image, param)
            if ellipsoids is None:
                input('No ellipsoids, press Enter to continue')
       
                continue

            if args.use_t265:
                ellipsoid_params_data = []  # List of ellipsoid params in world frame

                #T = t265_sub.get_world_camera_tf()
                
                                
                for i, (A, centroid) in enumerate(ellipsoids):
                    _, D, V = la.svd(A)
                    rx, ry, rz = 1.0 / np.sqrt(D)
                    print('centroid coords in camera frame')
                    print(centroid)

                    D435_to_ellipsoid = create_transform_matrix(rotation=V, translation=centroid)
                    tm.add_transform("D435", f"Ellipsoid{i}", D435_to_ellipsoid)

                    # ellipsoid_centroid_world = rigid_transform(centroid[np.newaxis,:], T)
                    # ellipsoid_rotation_world = ellipsoid_centroid_world # R.from_matrix(world_to_D435_mat[0:3, 0:3] @ V).as_quat()
                    
                    #ellipsoid_params_data.append({"centroid": list(ellipsoid_centroid_world), "rotation": list(ellipsoid_rotation_world), "axis": [rx, ry, rz]})
                #print(ellipsoid_params_data)
                # all_ellipsoids.append({frame_key:ellipsoid_params_data, 'pose' : T})

                #ax = tm.plot_frames_in("world", s=0.1)
                ax = tm.plot_frames_in("D435", s=0.1, whitelist=["D435","Ellipsoid0"])
                ax.set_xlim(-0.5, 0.5)
                ax.set_ylim(-0.5, 0.5)
                ax.set_zlim(-0.5, 0.5)
                plt.waitforbuttonpress(1)
                input('Waiting for input')
                ax.clear() 
                


                # import json
                # with open(f'data_files/ellipoids.json', 'w', encoding='utf-8') as f:
                #     json.dump(all_ellipsoids, f, ensure_ascii=False, indent=4)
            frame_key += 1

    else:
        loaded = np.load(args.load_rgbd)
        frames = list(loaded.keys())
        ellipsoid_data = []

        for frame_key in frames:
            try:
                im, depth = np.uint8(loaded[frame_key][:, :, :3]), loaded[frame_key][:, :, 3].astype(np.int16)
            except:
                break

            if args.load_segmentation:
                if frame_key in detections:
                    ellipsoids, detection = run_pipeline(im, depth, detections[frame_key])
                else:
                    continue
            else:
                ellipsoids, detection = run_pipeline(im, depth)
                if args.save_segmentation:
                    detections[frame_key] = detection

            ellipsoid_data.append(ellipsoids)

        if args.save_segmentation:
            pickle.dump(detections, open(f"{args.data_files}/detections.p", "wb"))

        pickle.dump(ellipsoid_data, open(f"{args.data_files}/ellipsoids.p", "wb"))

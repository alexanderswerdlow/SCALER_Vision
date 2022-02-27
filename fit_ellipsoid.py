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
from util import get_transformation
from aruco import get_d435_to_wall

T265_to_D435_mat = np.array(
    [
        [0.999968402, -0.006753626, -0.004188075, -0.015890727],
        [-0.006685408, -0.999848172, 0.016093893, 0.028273059],
        [-0.004296131, -0.016065384, -0.999861654, -0.009375589],
        [0, 0, 0, 1],
    ]
)

Wall_Frame_to_T265_mat = np.array(
    [
        [1, 0, 0, 0.4],  # Right is positive
        [0, 1, 0, 0],  # Up is positive
        [0, 0, 1, 0.81],  # Out of wall is positive
        [0, 0, 0, 1],
    ]
)

# Filtering params
min_volume = 5e-5
min_score = 0.95
min_axis_aligned_bounding_box_len = 0.075
min_detections = 5

min_dist_detection_clustering = 0.05
handhold_voxel_downsample = 0.01
fit_tolerance = 0.01


def rgbd_to_pcl(rgb_im, depth_im, param, vis=False):
    """
    Calibration:  [ 1280x720  p[655.67 358.885]  f[906.667 906.783]  Inverse Brown Conrady [0 0 0 0 0] ] [0.0, 0.0, 0.0, 0.0, 0.0]
    """
    im_rgb = o3d.geometry.Image(np.ascontiguousarray(rgb_im))
    im_depth = o3d.geometry.Image(np.ascontiguousarray(depth_im))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(im_rgb, im_depth, convert_rgb_to_intensity=False)
    intrinsic, extrinsic = param
    # Take intrinsics from numpy array use as params
    intrinsic = o3d.camera.PinholeCameraIntrinsic(int(intrinsic[0]), int(intrinsic[1]), *intrinsic[2:])
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, project_valid_depth_only=True)
    if extrinsic is not None:
        pcd.transform(extrinsic)  # Transform from D435 Frame to T265 Frame
    if vis:
        o3d.visualization.draw_geometries([pcd])
    return pcd


def segment_image(im):
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

    if args.verbose:
        print(f"Segmentation took {time.time() - start}")
        v = Visualizer(im[:, :, ::-1], metadata=train_metadata, scale=0.75, instance_mode=ColorMode.IMAGE)
        cv2.imwrite(f"output/climbnet-{frame_key}.png", v.draw_instance_predictions(results).get_image()[:, :, ::-1])

    return scores, boxes, masks


def cluster_and_fit(im, depth, param, scores, boxes, masks):
    """Create point cloud based on segmented masks, cluster, and fit ellipsoids"""

    start = time.time()
    pcds, rejected_masks = [], set()
    for idx, mask in enumerate(masks):
        if scores[idx] > min_score:
            detected_rgb_mask, detected_depth_mask = np.zeros_like(im), -np.ones_like(depth)
            detected_rgb_mask[mask], detected_depth_mask[mask] = im[mask], depth[mask]

            pcd = rgbd_to_pcl(detected_rgb_mask, detected_depth_mask, param, vis=False)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            if pcd.get_axis_aligned_bounding_box().get_extent().max() < min_axis_aligned_bounding_box_len:
                rejected_masks.add(idx)
            else:
                pcds.append(pcd)
        else:
            rejected_masks.add(idx)

    background_rgb, background_depth = im.copy(), depth.copy()  # Background images with detected regions removed
    for idx, mask in enumerate(masks):
        if idx not in rejected_masks:
            background_rgb[mask] = [0, 0, 0]
            background_depth[mask] = -1

    pcd_background = rgbd_to_pcl(background_rgb, background_depth, param, vis=False).voxel_down_sample(voxel_size=0.005)

    if args.verbose:
        print(f"Projecting + Filtering points for frame {frame_key} took {time.time() - start}")
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection="3d")

        points = np.asarray(pcd_background.points)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.asarray(pcd_background.colors), label=f"Background", alpha=0.65, s=0.5)

    if args.viz:
        vis.clear_geometries()
        pcd_background.paint_uniform_color([0.9, 0.1, 0.1])
        vis.add_geometry(pcd_background)

    ellipsoids = []
    for idx, cluster_pcd in enumerate(pcds):
        if args.viz:
            vis.add_geometry(cluster_pcd)

        cluster_pcd = cluster_pcd.voxel_down_sample(voxel_size=handhold_voxel_downsample)
        points = np.asarray(cluster_pcd.points)

        try:
            start = time.time()
            A, centroid = outer_ellipsoid.outer_ellipsoid_fit(points, tol=fit_tolerance)
            _, D, V = la.svd(A)
            axes = 1.0 / np.sqrt(D)
            if ((4 / 3) * np.pi * np.prod(axes)) > min_volume:
                ellipsoids.append((A, centroid, R.from_matrix(V).as_quat(), axes))

        except Exception as e:
            print(f"Fit failed for {idx}: {e}")
            continue

        if args.verbose:
            outer_ellipsoid.plot_ellipsoid(A, centroid, "green", ax)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.asarray(cluster_pcd.colors), label=f"{idx}-pos(cm):{centroid*100}-ax:{axes*100*2}", alpha=0.8, s=0.1)
            ax.text(*centroid, f"{idx}", size=15, zorder=3, color="red")
    
    print(f"Fit + Project for {frame_key} took {time.time() - start}")

    if args.verbose:
        ax.view_init(elev=270, azim=270)
        ax.set_xlim3d([-0.75, 0.75])
        ax.set_ylim3d([0.75, -0.75])
        ax.set_zlim3d([0, 1])
        plt.title(f"Detected {len(predictions)} handholds at pos: {trans*100}")
        plt.legend(loc="best")
        plt.savefig(f"output/3d-{frame_key}.png", dpi=500, bbox_inches="tight")
        ax.cla()
        plt.close()

    if args.viz:
        vis.poll_events()
        vis.update_renderer()

    return ellipsoids


def run_pipeline(color_image, depth_image, ir_image, intrinsic, trans, rot, detection=None):
    start = time.time()
    if detection:
        scores, boxes, masks = detection
    else:
        if (detection := segment_image(color_image)) is None:
            return None, None
        scores, boxes, masks = detection

    # For T265
    # extrinsic = get_transformation(trans, rot) @ T265_to_D435_mat

    # For Aruco Tags
    global d435_to_wall
    extrinsic = None
    if d435_to_wall is None:
        d435_to_wall, frame_aruco, aruco_centroid = get_d435_to_wall(color_image, intrinsic, trans, rot)
        if d435_to_wall is None:
            print("Failed to find aruco tag")
            return None, detection
    else:
        _, frame_aruco, aruco_centroid = get_d435_to_wall(color_image, intrinsic, trans, rot)

    plt.imsave(f'output/aruco-{frame_key}.png', frame_aruco)

    extrinsic = get_transformation(trans, rot) @ T265_to_D435_mat @ d435_to_wall # 

    if aruco_centroid is not None:
        detected_rgb_mask, detected_depth_mask = np.zeros_like(color_image), -np.ones_like(depth_image)
        aruco_centroid = aruco_centroid.mean(axis=0)
        detected_depth_mask[int(aruco_centroid[1]), int(aruco_centroid[0])] = depth_image[int(aruco_centroid[1]), int(aruco_centroid[0])]
        pcd_aruco = rgbd_to_pcl(detected_rgb_mask, detected_depth_mask, (intrinsic, extrinsic), vis=False)
        if len(np.asarray(pcd_aruco.points)) > 0:
            aruco_locs.append(np.asarray(pcd_aruco.points)[0][:2])

    ellipsoids = cluster_and_fit(color_image, depth_image, (intrinsic, extrinsic), scores, boxes, masks)
    
    if args.verbose:
        print(f"Frame {frame_key} took {time.time() - start}")
        plt.imsave(f"output/rgb-{frame_key}.png", color_image)

    for A, centroid, _, axes in ellipsoids:
        all_predictions.append((centroid, axes, int(frame_key)))
        found_match = False
        for idx, (center, A, pred_center, pred_axes) in enumerate(predictions):
            if np.abs(la.norm(center - centroid)) < min_dist_detection_clustering:
                predictions[idx][2].append(centroid)
                predictions[idx][3].append(axes)
                found_match = True
                break

        if not found_match:
            predictions.append((centroid, A, [centroid], [axes]))

    filtered_response = []
    if args.verbose:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim3d([-1, 1])
        ax.set_ylim3d([-0.5, 0.5])
        ax.set_zlim3d([0, 1])
        ax.view_init(elev=270, azim=270)

        for idx, (center, A, pred_center, pred_axes) in enumerate(predictions):
            if len(pred_center) > min_detections:
                centroid_predicted = np.mean(np.array([*pred_center]), axis=0)
                axes_predicted = np.mean(np.array([*pred_axes]), axis=0)
                centroid_predicted_std = np.std(np.array([*pred_center]), axis=0)
                axes_predicted_std = np.std(np.array([*pred_axes]), axis=0)
                filtered_response.append((A, centroid_predicted, axes_predicted, centroid_predicted_std, axes_predicted_std))
                outer_ellipsoid.plot_ellipsoid(A, centroid_predicted, "green", ax)
                print(idx, centroid_predicted_std*100)
    
        plt.title(f"Detected {len(predictions)} handholds at pos: {trans*100}")
        plt.savefig(f"output/map-{frame_key}.png", dpi=300, bbox_inches="tight")
        ax.cla()
        plt.close()

    return filtered_response, detection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ellipsoid Fitting")
    parser.add_argument("--load_segmentation", dest="load_segmentation", action="store_true")
    parser.add_argument("--save_segmentation", dest="save_segmentation", action="store_true")
    parser.add_argument("--use_t265_and_d435", dest="use_t265_and_d435", action="store_true")
    parser.add_argument("--run_from_file", dest="run_from_file", action="store_true")
    parser.add_argument("--viz", dest="viz", action="store_true")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--data_files", type=str, default="data_files")
    parser.add_argument("--capture", type=str, default="capture_1.npz")
    args = parser.parse_args()

    if args.load_segmentation:
        try:
            detections = pickle.load(open(f"{args.data_files}/segmentation/{args.capture.rstrip('.npz')}.p", "rb"))
        except (OSError, IOError) as e:
            print("detections.p not found, exiting")
            exit()
    else:
        import detectron2 as dt2
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.utils.visualizer import ColorMode, Visualizer
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor

        model_path = f"{args.data_files}/model_d2_R_50_FPN_3x.pth"
        dt2.data.datasets.register_coco_instances("climb_dataset", {}, f"{args.data_files}/mask.json", "")
        # model_path = f"{args.data_files}/training/model_final.pth"
        # dt2.data.datasets.register_coco_instances("climb_dataset", {}, f"datasets/coco/annotations/instances_empty.json", "")
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

    if args.viz:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    # if args.verbose:
    #     import os
    #     import shutil

    #     dir = 'output'
    #     if os.path.exists(dir):
    #         shutil.rmtree(dir)
    #     os.makedirs(dir)

    predictions = []
    all_predictions = []
    aruco_locs = []
    d435_to_wall = None
    if args.run_from_file:
        loaded = np.load(f"{args.data_files}/captures/{args.capture}", allow_pickle=True)
        frames = list(loaded.keys())
        print(len(frames))

        for frame_key in frames:
            if frame_key == "0":
                frame_key = "1"

            try:
                color_image, depth_image, ir_image, intrinsic, trans, rot = loaded[frame_key]
            except:
                print("Error, exiting")
                break

            prev_detection = detections[frame_key] if args.load_segmentation else None
            ellipsoids, detection = run_pipeline(color_image, depth_image, ir_image, intrinsic, trans, rot, prev_detection)
            if args.save_segmentation:
                detections[frame_key] = detection

            if ellipsoids is None:
                continue

        if args.save_segmentation:
            pickle.dump(detections, open(f"{args.data_files}/segmentation/{args.capture.rstrip('.npz')}.p", "wb"))
            exit()

        from scipy.spatial import KDTree
        in_to_m = 0.0254
        handhold_gt = np.array([[5, 0], [8, 4], [8, 9], [10, 12], [8, -4], [0, -5], [0, 5], [-6, -3], [-6, 3], [-17, -4], [-17, 2], [-14, 10]]).astype(np.float64)
        handhold_gt *= in_to_m
        handhold_tree = KDTree(handhold_gt)

        from collections import defaultdict
        handholds = defaultdict(list)
        plt.figure()
        for idx, (center, axes, frame_rec) in enumerate(all_predictions):
            dist, idx = handhold_tree.query(center[:2])
            handholds[idx].append((center, axes, frame_rec))
            plt.scatter(*(center[:2] * 39.3701))

        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.savefig('output/centroid.png')
        plt.close()

        plt.figure()
        for idx, (ar_loc) in enumerate(aruco_locs):
            plt.scatter(*(ar_loc * 100), s=35)

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.savefig('output/aruco_locs.png')
        plt.close()

        for idx, detected_instances in handholds.items():
            center_errors, centers = [], []
            for center, axes, frame_rec in detected_instances:
                center_errors.append(np.linalg.norm(handhold_gt[idx] - center[:2]))
                centers.append(center[:2])

            center_predicted_mean = np.mean(np.array([*center_errors]), axis=0)
            center_predicted_std = np.std(np.array([*center_errors]), axis=0)
            center_predicted_mean_ = np.mean(np.array([*centers]), axis=0)
            center_predicted_std_ = np.std(np.array([*centers]), axis=0)
            print(idx, center_predicted_mean*100, center_predicted_std*100, center_predicted_mean_*100, center_predicted_std_*100)

        breakpoint()

        if args.save_segmentation:
            pickle.dump(detections, open(f"{args.data_files}/generated/detections.p", "wb"))

    elif args.use_t265_and_d435:
        import d435_sub
        import t265_sub

        frame_key = 0
        all_ellipsoids = []
        while True:
            (color_image, depth_image, ir_image), (intrinsic, _) = d435_sub.get_rgbd()
            trans, rot = t265_sub.get_pose()
            ellipsoids, detection = run_pipeline(color_image, depth_image, ir_image, intrinsic, trans, rot)
            if ellipsoids is None:
                continue

            ellipsoid_params_data = []  # List of ellipsoid params in world frame
            for A, centroid, rotation, axes in ellipsoids:
                ellipsoid_params_data.append({"frame": frame_key, "centroid": list(centroid), "rotation": list(rotation), "axis": list(axes)})

            all_ellipsoids.append(ellipsoid_params_data)

            import json

            with open(f"data_files/ellipoids.json", "w", encoding="utf-8") as f:
                json.dump(all_ellipsoids, f, ensure_ascii=False, indent=4)
            frame_key += 1

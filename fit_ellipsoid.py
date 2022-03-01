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
from util import get_transformation, get_mean_std, T265_to_D435_mat, camera_intrinsics, camera_distortion, camera_width, camera_height
from aruco import get_d435_to_wall
import json

# Filtering params
min_volume = 5e-5
min_score = 0.95
min_axis_aligned_bounding_box_len = 0.075
min_detections = 5

min_dist_detection_clustering = 0.05
handhold_voxel_downsample = 0.001
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
    intrinsic = o3d.camera.PinholeCameraIntrinsic(camera_width, camera_height, camera_intrinsics[0,0], camera_intrinsics[1,1], camera_intrinsics[0, 2], camera_intrinsics[1, 2])
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

def border_elems_generic(a, W): # Input array : a, Edgewidth : W
    n1 = a.shape[0]
    r1 = np.minimum(np.arange(n1)[::-1], np.arange(n1))
    n2 = a.shape[1]
    r2 = np.minimum(np.arange(n2)[::-1], np.arange(n2))
    return a[np.minimum(r1[:,None],r2)<W]

def cluster_and_fit(im, depth, param, scores, boxes, masks):
    """Create point cloud based on segmented masks, cluster, and fit ellipsoids"""

    start = time.time()
    pcds, rejected_masks = [], set()
    for idx, mask in enumerate(masks):
        if scores[idx] > min_score:
            detected_rgb_mask, detected_depth_mask = np.zeros_like(im), -np.ones_like(depth)
            detected_rgb_mask[mask], detected_depth_mask[mask] = im[mask], depth[mask]

            pcd = rgbd_to_pcl(detected_rgb_mask, detected_depth_mask, param, vis=False)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.1)

            if pcd.get_axis_aligned_bounding_box().get_extent().max() < min_axis_aligned_bounding_box_len or np.any(border_elems_generic(mask, 1)):
                rejected_masks.add(idx)
            else:
                pcds.append(pcd)
        else:
            rejected_masks.add(idx)

    if args.verbose:
        background_rgb, background_depth = im.copy(), depth.copy()  # Background images with detected regions removed
        for idx, mask in enumerate(masks):
            if idx not in rejected_masks:
                background_rgb[mask] = [0, 0, 0]
                background_depth[mask] = -1

        pcd_background = rgbd_to_pcl(background_rgb, background_depth, param, vis=False) #.voxel_down_sample(voxel_size=0.005)
        print(f"Projecting + Filtering points for frame {frame_key} took {time.time() - start}")
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection="3d")

        points = np.asarray(pcd_background.points)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.asarray(pcd_background.colors), label=f"Background", alpha=0.65, s=0.1)

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
        # plt.title(f"Detected {len(ellipsoids)} handholds at pos: {trans*100}")
        # plt.legend(loc="best")
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

    print(f"Frame Segmentation {frame_key} took {time.time() - start}")
    use_aruco = False
    if use_aruco:
        # For Aruco Tags
        global d435_to_wall
        extrinsic = None
        if d435_to_wall is None:
            d435_to_wall, _ = get_d435_to_wall(color_image, intrinsic, trans, rot, frame_key)
            if d435_to_wall is None:
                print("Failed to find aruco tag")
                return None, detection
        else:
            get_d435_to_wall(color_image, intrinsic, trans, rot, frame_key)

        extrinsic = d435_to_wall @ get_transformation(trans, rot) @ T265_to_D435_mat
    else:
        # For T265 Only
        extrinsic = get_transformation(trans, rot) @ T265_to_D435_mat
    print(f"Frame Before Cluster/Fit {frame_key} took {time.time() - start}")
    ellipsoids = cluster_and_fit(color_image, depth_image, (intrinsic, extrinsic), scores, boxes, masks)
    print(f"Frame {frame_key} took {time.time() - start}")
    
    if args.verbose:
        
        plt.imsave(f"output/rgb-{frame_key}.png", color_image)

    for A, centroid, _, axes in ellipsoids:
        all_predictions.append((centroid, axes, int(frame_key)))

    filtered_response = []
    handholds.clear()
    for idx, (center, axes, frame_rec) in enumerate(all_predictions):
        dist, idx = handhold_tree.query(center[:2])
        if dist < 0.05:
            handholds[idx].append((center, axes, frame_rec))

    all_err, all_euc_err, all_centers = [], [], []
    for idx, detected_instances in handholds.items():
        center_errors, centers = [], []
        for center, axes, frame_rec in detected_instances:
            center_errors.append(np.linalg.norm(handhold_gt[idx] - center[:2])*100)
            all_err.append((handhold_gt[idx] - center[:2])*100)
            centers.append(center[:2]*100)
            all_centers.append(center[:2]*100)
            all_euc_err.append(np.linalg.norm(handhold_gt[idx] - center[:2])*100)

        # if args.verbose:
        #     outer_ellipsoid.plot_ellipsoid(A, centroid, "green", ax)
        #     ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.asarray(cluster_pcd.colors), label=f"{idx}-pos(cm):{centroid*100}-ax:{axes*100*2}", alpha=0.8, s=0.1)
        #     ax.text(*centroid, f"{idx}", size=15, zorder=3, color="red")

        print(f'GT for {idx}: {100*handhold_gt[idx]}, Center: {get_mean_std(centers)}, Err: {get_mean_std(center_errors)}')

    if len(handholds) > 0:  
        print(f"GT Overall Err: Euclidean: {get_mean_std(all_euc_err)}, X,Y: {get_mean_std(all_err)}")

    # if args.verbose:
    #     fig = plt.figure(figsize=(12, 12))
    #     ax = fig.add_subplot(111, projection="3d")


    #     ax.set_xlim3d([-1, 1])
    #     ax.set_ylim3d([-0.5, 0.5])
    #     ax.set_zlim3d([0, 1])
    #     ax.view_init(elev=270, azim=270)
    #     plt.title(f"Detected {len(ellipsoids)} handholds at pos: {trans*100}")
    #     plt.savefig(f"output/map-{frame_key}.png", dpi=300, bbox_inches="tight")
    #     ax.cla()
    #     plt.close()

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

    if args.verbose:
        import os
        import shutil

        dir = 'output'
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

    all_predictions = []
    d435_to_wall = None
    from scipy.spatial import KDTree
    in_to_m = 0.0254
    handhold_gt = np.array([[5, 0], [8, 4], [8, 9], [10, 12], [8, -4], [0, -5], [0, 5], [-6, -3], [-6, 3], [-17, -4], [-17, 2], [-14, 10]]).astype(np.float64)
    handhold_gt *= in_to_m
    handhold_tree = KDTree(handhold_gt)

    from collections import defaultdict
    handholds = defaultdict(list)
    if args.run_from_file:
        loaded = np.load(f"{args.data_files}/captures/{args.capture}", allow_pickle=True)
        frames = list(loaded.keys())
        print(len(frames))

        for frame_key in frames:

            frame_idx = int(frame_key)

            # if frame_idx < 0 or frame_idx > 60:
            #     continue

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

        print(f"Saving predictions as {args.data_files}/generated/{args.capture.rstrip('.npz')}_predictions.p")
        with open(f"{args.data_files}/generated/{args.capture.rstrip('.npz')}_predictions.p", 'wb') as handle:
            pickle.dump(all_predictions, handle)

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

            with open(f"data_files/ellipoids.json", "w", encoding="utf-8") as f:
                json.dump(all_ellipsoids, f, ensure_ascii=False, indent=4)
            frame_key += 1

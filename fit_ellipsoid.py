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

T265_to_D435_mat = np.array([
        [0.999968402, -0.006753626, -0.004188075, -0.015890727],
        [-0.006685408, -0.999848172, 0.016093893, 0.028273059],
        [-0.004296131, -0.016065384, -0.999861654, -0.009375589],
        [0, 0, 0, 1]
        ])

min_volume = 5e-5
min_score = 0.80

def rgbd_to_pcl(rgb_im, depth_im, param, vis=False):
    """
    Calibration:  [ 1280x720  p[655.67 358.885]  f[906.667 906.783]  Inverse Brown Conrady [0 0 0 0 0] ] [0.0, 0.0, 0.0, 0.0, 0.0]
    """
    im_rgb = o3d.geometry.Image(np.ascontiguousarray(rgb_im))
    im_depth = o3d.geometry.Image(np.ascontiguousarray(depth_im))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(im_rgb, im_depth, convert_rgb_to_intensity=False)
    intrinsic, extrinsic = param
    intrinsic = o3d.camera.PinholeCameraIntrinsic(int(intrinsic[0]), int(intrinsic[1]), *intrinsic[2:])
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, project_valid_depth_only=True)
    if extrinsic is not None:
        pcd.transform(extrinsic)
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
        v = Visualizer(
            im[:, :, ::-1],
            metadata=train_metadata,
            scale=0.75,
            instance_mode=ColorMode.IMAGE,
        )
        cv2.imwrite(f"output/{frame_key}-climbnet.jpg", v.draw_instance_predictions(results).get_image()[:, :, ::-1])

    return scores, boxes, masks


def cluster_and_fit(im, depth, param, scores, boxes, masks):
    """Create point cloud based on segmented masks, cluster, and fit ellipsoids"""
    # Empty images to put detected regions
    detected_rgb, detected_depth = np.zeros_like(im), -np.ones_like(depth)
    for idx, mask in enumerate(masks):
        if scores[idx] > min_score:
            detected_rgb[mask] = im[mask]
            detected_depth[mask] = depth[mask]

    # Generate point cloud from RGBD
    pcd = rgbd_to_pcl(detected_rgb, detected_depth, param, vis=False)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if len(pcd.points) == 0:
        return None

    # Segment point cloud, returning (n,) array of labels
    start = time.time()
    labels = np.array(pcd.cluster_dbscan(eps=0.004, min_points=10, print_progress=False))

    if args.verbose:
        print(f"Clustering points for frame {frame_key} took {time.time() - start}")
        background_rgb, background_depth = im.copy(), depth.copy()  # Background images with detected regions removed
        for idx, mask in enumerate(masks):
            if scores[idx] > min_score:
                background_rgb[mask] = [0, 0, 0]
                background_depth[mask] = -1

        fig = plt.figure(figsize=(8.0, 8.0))
        ax = fig.add_subplot(111, projection="3d")

        pcd_background = rgbd_to_pcl(background_rgb, background_depth, param, vis=False).voxel_down_sample(voxel_size=0.01)
        points = np.asarray(pcd_background.points)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.asarray(pcd_background.colors), label=f"Background", alpha=0.1, s=0.5)

    if args.vis:
        vis.clear_geometries()

    ellipsoids = []
    for idx in range(labels.max() + 1):
        cluster_pcd = pcd.select_by_index(np.argwhere(labels == idx))
        if args.vis:
            vis.add_geometry(cluster_pcd)

        if cluster_pcd.get_axis_aligned_bounding_box().get_extent().max() < 0.05:
            continue

        cluster_pcd = cluster_pcd.voxel_down_sample(voxel_size=0.005)
        points = np.asarray(cluster_pcd.points)

        try:
            start = time.time()
            A, centroid = outer_ellipsoid.outer_ellipsoid_fit(points, tol=0.01)
            _, D, V = la.svd(A)
            axes = 1.0 / np.sqrt(D)
            if ((4 / 3) * np.pi * np.prod(axes)) > min_volume:
                ellipsoids.append((A, centroid, R.from_matrix(V).as_quat(), axes))

        except Exception as e:
            print(f"Fit failed for {idx}: {e}")
            continue

        if args.verbose:
            print(f"Fit for {idx} took {time.time() - start} for {len(points)} points")
            outer_ellipsoid.plot_ellipsoid(A, centroid, "green", ax)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.asarray(cluster_pcd.colors), label=f"{idx}")

    if args.verbose:
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
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        # ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        # ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        # ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

        ax.set_xlim3d([-0.5, 0.5])
        ax.set_ylim3d([-0.5, 0.5])
        ax.set_zlim3d([0, 1])
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
        if (detection := segment_image(color_image)) is None:
            return None, None
        scores, boxes, masks = detection

    ellipsoids = cluster_and_fit(color_image, depth_image, param, scores, boxes, masks)
    if args.verbose:
        print(f"Frame {frame_key} took {time.time() - start}")
    return ellipsoids, detection


def get_transformation(trans, rot):
    r = R.from_quat(rot)
    rotation = np.array(r.as_matrix())
    translation = trans[np.newaxis].T
    T = np.hstack((rotation, translation))
    T = np.vstack((T, np.array([0, 0, 0, 1])))
    return T

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ellipsoid Fitting")
    parser.add_argument("--load_segmentation", dest="load_segmentation", action="store_true")
    parser.add_argument("--save_segmentation", dest="save_segmentation", action="store_true")
    parser.add_argument("--use_t265_and_d435", dest="use_t265_and_d435", action="store_true")
    parser.add_argument("--run_from_file", dest="run_from_file", action="store_true")
    parser.add_argument("--vis", dest="vis", action="store_true")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--data_files", type=str, default="data_files")
    args = parser.parse_args()

    if args.load_segmentation:
        try:
            detections = pickle.load(open(f"{args.data_files}/detections.p", "rb"))
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
        #model_path = f"{args.data_files}/training/model_final.pth"
        #dt2.data.datasets.register_coco_instances("climb_dataset", {}, f"datasets/coco/annotations/instances_empty.json", "")
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
        loaded = np.load(f"{args.data_files}/capture_1.npz", allow_pickle=True)
        frames = list(loaded.keys())
        predictions = []

        for frame_key in frames[::2]:
            try:
                color_image, depth_image, ir_image, intrinsic, trans, rot = loaded[frame_key]
            except:
                print('Error, exiting')
                break

            extrinsic = get_transformation(trans, rot) @ T265_to_D435_mat
            ellipsoids, detection = run_pipeline(color_image, depth_image, (intrinsic, extrinsic), detections[frame_key] if args.load_segmentation else None)
            if args.save_segmentation:
                detections[frame_key] = detection
            if ellipsoids is None:
                print("None!")
                continue

            for A, centroid, rotation, axes in ellipsoids:
                found_match = False
                for idx, (center, A, pred_center, pred_axes) in enumerate(predictions):
                    if np.abs(np.linalg.norm(center - centroid)) < 0.05:
                        predictions[idx][2].append(centroid)
                        predictions[idx][3].append(axes)
                        found_match = True
                        break

                if not found_match:
                    predictions.append((centroid, A, [centroid], [axes]))
        
        fig = plt.figure(figsize=(8.0, 8.0))
        ax = fig.add_subplot(111, projection="3d")
        for idx, (center, A, pred_center, pred_axes) in enumerate(predictions):
            if len(pred_center) > 10:
                centroid_predicted = np.mean(np.array([*pred_center]), axis=0)
                outer_ellipsoid.plot_ellipsoid(A, centroid_predicted, "green", ax)

        ax.set_xlim3d([-1, 1])
        ax.set_ylim3d([-1, 1])
        ax.set_zlim3d([0, 2])
        ax.view_init(elev=270, azim=270)
        plt.legend(loc="best")
        plt.savefig(f"output/all_detections.jpg", dpi=300, bbox_inches="tight")
        plt.close()
        
    elif args.use_t265_and_d435:
        import d435_sub
        import t265_sub
        
        frame_key = 0
        all_ellipsoids = []
        while True:
            if frame_key < 10:
                frame_key += 1
                continue
            input("Press any key to take picture")
            (color_image, depth_image, _), (intrinsic, _) = d435_sub.get_rgbd()
            trans, rot = t265_sub.get_pose()
            extrinsic = get_transformation(trans, rot) @ T265_to_D435_mat
            ellipsoids, detection = run_pipeline(color_image, depth_image, (intrinsic, extrinsic))
            if ellipsoids is None:
                print("None!")
                continue
            
            ellipsoid_params_data = []  # List of ellipsoid params in world frame
            for A, centroid, rotation, axes in ellipsoids:
                ellipsoid_params_data.append({"frame" : frame_key, "centroid": list(centroid), "rotation": list(rotation), "axis": list(axes)})
                print(centroid)
            
            all_ellipsoids.append(ellipsoid_params_data)

            import json
            with open(f'data_files/ellipoids.json', 'w', encoding='utf-8') as f:
                json.dump(all_ellipsoids, f, ensure_ascii=False, indent=4)
            frame_key += 1

    else:
        loaded = np.load(f'{args.data_files}/rgbd.npz')
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

        pickle.dump(ellipsoid_data, open(f"{args.data_files}/ellipsoids.p", "wb"))

    if args.save_segmentation:
        pickle.dump(detections, open(f"{args.data_files}/detections.p", "wb"))

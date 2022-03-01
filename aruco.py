import numpy as np
import cv2
from util import rvec_2_euler, get_transformation
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from util import camera_distortion, camera_intrinsics
np.set_printoptions(precision=3, suppress=True)

def view():
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    ir_colormap = cv2.applyColorMap(cv2.convertScaleAbs(ir_image, alpha=0.03), cv2.COLORMAP_JET)
    images = np.hstack((color_image, depth_colormap, ir_colormap))

    # Show images
    cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("RealSense", images)
    cv2.waitKey(1)

def get_multi_tags(img, frame_key):
    draw_img = img.copy()
    from sklearn.neighbors import KDTree
    

    tag_size_pixels = 1000

    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    arucoParams.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    arucoParams.cornerRefinementWinSize = 25
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (aruco_tag_corners, ids, rejected) = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)
    aruco_tag_corners = np.array(aruco_tag_corners)

    def find_holomography(corners, tag_size_pixels):
        src_pts = corners
        dst_pts = np.array([[0, tag_size_pixels], [tag_size_pixels, tag_size_pixels], [tag_size_pixels, 0], [0, 0]])

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M


    def detect_features(image, max_num_features=500, threshold=0.001, min_distance=3):
        detected_corners = cv2.goodFeaturesToTrack(image, 1000, 0.001, 2)
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
        # Calculate the refined corner locations
        detected_corners = cv2.cornerSubPix(image, detected_corners, winSize, zeroZone, criteria)
        return detected_corners


    def load_tag(tag_number, tag_size_pixels):
        arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)  # Fixed typo
        tag = np.zeros((tag_size_pixels, tag_size_pixels, 1), dtype="uint8")
        cv2.aruco.drawMarker(arucoDict, tag_number, tag_size_pixels, tag, 1)
        tag = tag.squeeze()
        return tag


    def match_features(detected_features_image_transformed, detected_features_tag):
        X = detected_features_image_transformed.squeeze()  # 10 points in 3 dimensions
        tree = KDTree(X, leaf_size=2)

        detected_features_tag = detected_features_tag.squeeze()

        points_3d = []
        points_2d = []

        for i in range(detected_features_tag.shape[0]):
            dist, ind = tree.query(detected_features_tag[i].reshape((1, -1)), k=1)
            points_3d.append(detected_features_tag[i])
            points_2d.append(detected_features_image[int(ind)])

        points_3d = np.array(points_3d)
        points_2d = np.array(points_2d)
        points_2d = points_2d.squeeze()

        points_3d /= tag_size_pixels

        return (points_2d, points_3d)


    should_plot = False
    offsets = {0: np.array([-14, 0]), 1: np.array([-7, 6]), 3: np.array([0, -2]), 4: np.array([12, 5])}
    points_3d_all, points_2d_all = [], []
    if ids is None:
        return np.zeros(3), np.zeros(4)
    for tag_index in range(len(ids)):
        # cv2.aruco.drawDetectedMarkers(draw_img, aruco_tag_corners[tag_index].reshape(4, 2))
        tag_number = int(ids[tag_index])

        M = find_holomography(aruco_tag_corners[tag_index, :, :], tag_size_pixels)

        detected_features_image = detect_features(gray)

        if should_plot:
            plt.rcParams["figure.figsize"] = [20, 20]
            for i in detected_features_image:
                x, y = i.ravel()
                plt.scatter(x, y)
            plt.imshow(img[:, :, ::-1])
            plt.savefig(f"plot-{tag_index}.png")

        detected_features_image_transformed = cv2.perspectiveTransform(detected_features_image, M)
        detected_features_image_transformed[:, :, 1] = tag_size_pixels - detected_features_image_transformed[:, :, 1]
        tag = load_tag(tag_number, tag_size_pixels)
        detected_features_tag = detect_features(tag, max_num_features=500, threshold=0.1, min_distance=50)
        detected_features_tag[:, :, 1] = tag_size_pixels - detected_features_tag[:, :, 1]

        if should_plot:
            for feature in detected_features_image_transformed:
                x, y = feature.ravel()
                plt.scatter(x, y, color="r", alpha=0.5, s=1_000)

            for feature in detected_features_tag:
                x, y = feature.ravel()
                plt.scatter(x, y, color="b", alpha=0.5, s=1_000)

            plt.imshow(tag, cmap="gray")
            plt.savefig(f"plot2-{tag_index}.png")

        points_2d_interior, points_3d_interior = match_features(detected_features_image_transformed, detected_features_tag)

        points_3d_exterior = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])

        points_2d_exterior = aruco_tag_corners[tag_index, :, :].squeeze()

        points_3d = np.vstack([points_3d_exterior, points_3d_interior])
        points_2d = np.vstack([points_2d_exterior, points_2d_interior])
        points_3d = np.hstack([points_3d, np.zeros((points_3d.shape[0], 1))]) * 0.1

        if should_plot:
            plt.scatter(points_3d[:, 0], points_3d[:, 1])
            plt.savefig(f"plot3-{tag_index}.png")

        points_3d[:, :2] += (offsets[tag_number] * 0.0254) + np.array([0, -16 * 0.0254])
        points_3d_all.append(points_3d)
        points_2d_all.append(points_2d)


    points_3d_global, points_2d_global = points_3d_all.pop(), points_2d_all.pop()

    for i, j in zip(points_3d_all, points_2d_all):
        points_3d_global = np.vstack([points_3d_global, i])
        points_2d_global = np.vstack([points_2d_global, j])

    if should_plot:
        plt.figure()
        plt.scatter(points_3d_global[:, 0]*39.37007874, points_3d_global[:, 1]*39.37007874)
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        plt.savefig(f"plot_all.png")

    
    ret, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=points_3d, imagePoints=points_2d, cameraMatrix=camera_intrinsics, distCoeffs=camera_distortion, reprojectionError=2, iterationsCount=3000)

    cv2.aruco.drawAxis(draw_img, camera_intrinsics, camera_distortion, rvec, tvec, 5 * 0.0254)
    plt.imsave(f'output/axis-{frame_key}.png', draw_img)


    rot_aruco = R.from_rotvec(rvec[:, 0]).as_quat()
    t = np.array(tvec).flatten()
    return t, rot_aruco

def get_d435_to_wall(frame, intrinsics, trans, rot, frame_key, draw_frame=True):

    """
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    """

    # General
    # tvec = tvec.flatten() * 10
    # tvec[0] *= -1
    # tvec[1] *= -1
    # print(tvec, trans)
    # rvec = R.from_rotvec(np.zeros(3)).as_quat()
    # d435_to_wall = get_transformation(tvec, rvec)

    # 2022_02_26-06_52_39_PM.npz
    # tvec = np.array([-0.01989264 + (0.0037),  -0.02225 + (-0.009), 0])
    # rvec = R.from_rotvec(np.zeros(3)).as_quat()
    # d435_to_wall = get_transformation(tvec, rvec)

    # 2022_02_26-03_37_03_PM.npz
    # tvec = np.array([[ 0.5],[0],[ -0.83944742]])
    # rvec = np.array([-0.00300064,  0.09308793, -0.00868328,  0.9956155 ])
    # transformer = np.eye(4)
    # transformer[2,2] = -1
    # transformer[3,3] = 1
    # d435_to_wall = get_transformation(tvec.flatten(), rvec.flatten()) @ transformer

    # TODO: Something with the rotation needs to be flipped, either in rvec, transformer or both
    # 2022_02_27-09_48_26_PM.npz
    tvec, rot_aruco = get_multi_tags(frame.copy(), frame_key)
    tvec_init = np.array([0.066, 0.292, 1.01 ])
    transformer = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

    T265_to_D435_mat = np.array(
        [
            [0.999968402, -0.006753626, -0.004188075, -0.015890727],
            [-0.006685408, -0.999848172, 0.016093893, 0.028273059],
            [-0.004296131, -0.016065384, -0.999861654, -0.009375589],
            [0, 0, 0, 1],
        ]
    )

    tvec[1] += 5.5 * 0.0254
    print(tvec, np.linalg.norm((tvec_init - tvec) - trans))
    try:
        d435_to_wall = get_transformation(tvec.flatten(), rot_aruco.flatten()) @ transformer @ np.linalg.inv(T265_to_D435_mat) @ np.linalg.inv(get_transformation(trans, rot))
    except:
        d435_to_wall = None

    # 2/27 5:43 - 7PM
    # in_to_m = 0.0254
    # mm_to_m = 0.001
    # tvec = -np.array(
    #     [-4 * in_to_m + 2 * in_to_m + (-12.54 * mm_to_m) + (25.62 * mm_to_m), 
    #     (-0 * in_to_m) + (-20 * in_to_m) + (12.54 * mm_to_m) + (5.3 * mm_to_m) + (12.25 * mm_to_m), 
    #     0.46 + (-5.3 * mm_to_m)]
    # )
    # rvec = R.from_rotvec(np.zeros(3)).as_quat()
    # d435_to_wall = get_transformation(tvec, rvec)

    return d435_to_wall, frame


if __name__ == "__main__":
    import d435_sub
    from datetime import datetime

    current_date_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    idx = 0
    try:
        while True:
            (color_image, depth_image, ir_image), (intrinsics, extrinsic) = d435_sub.get_rgbd()
            if color_image is None:
                continue

            d435_to_wall = get_d435_to_wall(color_image, intrinsics)

            cv2.imshow("Estimated Pose", color_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            idx += 1

    finally:
        cv2.destroyAllWindows()

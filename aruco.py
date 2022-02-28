import numpy as np
import cv2
from util import rvec_2_euler, get_transformation
from scipy.spatial.transform import Rotation as R


def view():
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    ir_colormap = cv2.applyColorMap(cv2.convertScaleAbs(ir_image, alpha=0.03), cv2.COLORMAP_JET)
    images = np.hstack((color_image, depth_colormap, ir_colormap))

    # Show images
    cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("RealSense", images)
    cv2.waitKey(1)


def get_d435_to_wall(frame, intrinsics, trans, rot, draw_frame=True):

    """
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    """

    if draw_frame:
        frame = np.copy(frame)

    aruco_dict_type = cv2.aruco.DICT_5X5_1000
    matrix_coefficients = np.array([[intrinsics[2], 0, intrinsics[4]], [0, intrinsics[3], intrinsics[5]], [0, 0, 1]])
    distortion_coefficients = np.zeros((1, 5))

    matrix_coefficients = np.array([[882.63940102, 0.0, 637.29136233], [0.0, 885.00533948, 365.28868425], [0.0, 0.0, 1.0]])

    distortion_coefficients = np.array([[0.12041179, -0.38617633, 0.00104598, -0.00789058, 0.35420179]])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, _ = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters, cameraMatrix=matrix_coefficients, distCoeff=distortion_coefficients)
    tvec, rvec = None, None
    centroid_aruco = None

    if len(corners) > 0:  # If markers are detected
        centroid_aruco = corners[0].mean(axis=0)
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            # The marker corrdinate system is centered on the middle of the marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.01, matrix_coefficients, distortion_coefficients)

            if draw_frame:
                cv2.aruco.drawDetectedMarkers(frame, corners)  # Draw a square around the markers
                cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis

    

    use_rot = False

    if use_rot:
        tvec = -tvec.flatten() * 10
        tvec[1] *= -1

        if rvec is not None:
            rvec = rvec.flatten()
            rvec = R.from_rotvec(rvec).as_quat()

        # euler_rvec = R.from_rotvec(rvec.flatten()).as_euler("xyz").flatten()
        # euler_rvec[0] *= -1
        # quat_rvec = R.from_euler("xyz", euler_rvec).as_quat()
        flip_yz = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        d435_to_wall = get_transformation(tvec, rvec) @ flip_yz

    else:
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
        # From Aruco Data in D435 Frame
        tvec = -np.array([[0.062],[-0.238],[1.064]])
        rvec = np.array([-0.066,  0.055, -0.009,  0.996])
        rvec = R.from_quat(rvec).as_euler('xyz')
        rvec[0] *= 1
        rvec[1] *= 1
        rvec[2] *= 1
        rvec = R.from_euler('xyz', rvec).as_quat()


        transformer = np.eye(4)
        transformer[0,0] = 1
        transformer[1,1] = -1
        transformer[2,2] = 1

        T265_to_D435_mat = np.array(
            [
                [0.999968402, -0.006753626, -0.004188075, -0.015890727],
                [-0.006685408, -0.999848172, 0.016093893, 0.028273059],
                [-0.004296131, -0.016065384, -0.999861654, -0.009375589],
                [0, 0, 0, 1],
            ]
        )


        

        d435_to_wall = get_transformation(tvec.flatten(), rvec.flatten()) @ transformer @ np.linalg.inv(T265_to_D435_mat) @ np.linalg.inv(get_transformation(trans, rot))


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

    return d435_to_wall, frame, centroid_aruco


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

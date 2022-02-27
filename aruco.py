import numpy as np
import pyrealsense2 as rs
import cv2
from util import rvec_2_euler, get_transformation


def view():
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    ir_colormap = cv2.applyColorMap(cv2.convertScaleAbs(ir_image, alpha=0.03), cv2.COLORMAP_JET)
    images = np.hstack((color_image, depth_colormap, ir_colormap))

    # Show images
    cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("RealSense", images)
    cv2.waitKey(1)


def get_d435_to_wall(frame, intrinsics, rot, draw_frame=False):

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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, _ = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters, cameraMatrix=matrix_coefficients, distCoeff=distortion_coefficients)
    tvec, rvec = None, None

    if len(corners) > 0:  # If markers are detected
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            # The marker corrdinate system is centered on the middle of the marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.01, matrix_coefficients, distortion_coefficients)

            if draw_frame:
                cv2.aruco.drawDetectedMarkers(frame, corners)  # Draw a square around the markers
                cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis

    if rvec is not None:

        use_rot = False

        if use_rot:
            from scipy.spatial.transform import Rotation as R

            tvec = -tvec.flatten() * 10
            tvec[1] *= -1

            # euler_rvec = R.from_rotvec(rvec.flatten()).as_euler("xyz").flatten()
            # euler_rvec[2] *= -1
            # quat_rvec = R.from_euler("xyz", euler_rvec).as_quat()

            d435_to_wall = get_transformation(tvec, rot)
            # breakpoint()
        else:
            tvec = -tvec.flatten() * 10
            tvec[1] *= -1
            rvec = rvec_2_euler(np.zeros(3)).flatten()
            d435_to_wall = get_transformation(tvec, rvec)

        #breakpoint()
    else:
        d435_to_wall = None

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
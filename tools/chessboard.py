    
import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import d435_sub

def calibrate(square_size=0.022):
    """ Apply camera calibration operation for images in the given directory path.
        Credit: aliyasineser"""
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    width = 9
    height = 6

    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # data_files/captures/2022_02_27-03_28_31_PM.npz
    # data_files/captures/2022_02_27-03_27_19_PM.npz
    # loaded = np.load(f"data_files/captures/2022_02_27-03_28_31_PM.npz", allow_pickle=True)
    # frames = list(loaded.keys())
    # print(len(frames))

    for i in range(30):
        test = input('press to take picture')
        img, depth_image, ir_image, intrinsic, trans, rot = d435_sub.get_rgbd()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            #cv2.imshow(ARUCO.FRAME, img)


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(mtx, dist)

calibrate()


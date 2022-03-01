import zipfile
import io
import typing
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R

np.set_printoptions(suppress=True, precision=3)

T265_to_D435_mat = np.array(
    [
        [0.999968402, -0.006753626, -0.004188075, -0.015890727],
        [-0.006685408, -0.999848172, 0.016093893, 0.028273059],
        [-0.004296131, -0.016065384, -0.999861654, -0.009375589],
        [0, 0, 0, 1],
    ]
)

camera_width, camera_height = 1280, 720
# camera_intrinsics = np.array([[886.50658842, 0.0, 643.11152258], [0.0, 889.00345804, 363.11086262], [0.0, 0.0, 1.0]])
# camera_distortion = np.array([[0.12163025, -0.35153439, 0.00296531, -0.00498172, 0.27180912]])

# camera_intrinsics = np.array([[906.667, 0.0, 906.783], [0.0, 889.00345804, 358.885], [0.0, 0.0, 1.0]])
# camera_distortion = np.zeros(5)

# 03/01
camera_intrinsics = np.array([[897.76995141, 0.0, 643.26778658], [0.0, 900.99028667, 352.81364263], [0.0, 0.0, 1.0]])
camera_distortion = np.array([[ 0.1446363 , -0.41231416, -0.00208682, -0.00456624,  0.33925124]])

def get_mean_std(arr):
    return f"Mean: {np.mean(np.array([*arr]), axis=0)}, Std: {np.std(np.array([*arr]), axis=0)}"


def rvec_2_euler(rvec):
    EULER = "zyx"
    euler_rvec = R.from_rotvec(rvec).as_quat()
    return euler_rvec


def get_transformation(trans, rot):
    # breakpoint()
    rot = R.from_quat(rot).as_matrix()
    trans = trans[np.newaxis].T
    return np.vstack((np.hstack((rot, trans)), np.array([0, 0, 0, 1])))


class IncrementalNpzWriter:
    """
    Write data to npz file incrementally rather than compute all and write
    once, as in ``np.save``. This class can be used with ``contextlib.closing``
    to ensure closed after usage.
    """

    def __init__(self, tofile: str, mode: str = "x"):
        """
        :param tofile: the ``npz`` file to write
        :param mode: must be one of {'x', 'w', 'a'}. See
               https://docs.python.org/3/library/zipfile.html for detail
        """
        assert mode in "xwa", str(mode)
        self.tofile = zipfile.ZipFile(tofile, mode=mode, compression=zipfile.ZIP_DEFLATED)

    def write(self, key: str, data: typing.Union[np.ndarray, bytes], is_npy_data: bool = True) -> None:
        """
        :param key: the name of data to write
        :param data: the data
        :param is_npy_data: if ``True``, ".npz" will be appended to ``key``,
               and ``data`` will be serialized by ``np.save``;
               otherwise, ``key`` will be treated as is, and ``data`` will be
               treated as binary data
        :raise KeyError: if the transformed ``key`` (as per ``is_npy_data``)
               already exists in ``self.tofile``
        """
        if key in self.tofile.namelist():
            raise KeyError('Duplicate key "{}" already exists in "{}"'.format(key, self.tofile.filename))
        self.update(key, data, is_npy_data=is_npy_data)

    def update(self, key: str, data: typing.Union[np.ndarray, bytes], is_npy_data: bool = True) -> None:
        """
        Same as ``self.write`` but overwrite existing data of name ``key``.

        :param key: the name of data to write
        :param data: the data
        :param is_npy_data: if ``True``, ".npz" will be appended to ``key``,
               and ``data`` will be serialized by ``np.save``;
               otherwise, ``key`` will be treated as is, and ``data`` will be
               treated as binary data
        """
        kwargs = {
            "mode": "w",
            "force_zip64": True,
        }
        if is_npy_data:
            key += ".npy"
            with io.BytesIO() as cbuf:
                np.save(cbuf, data)
                cbuf.seek(0)
                with self.tofile.open(key, **kwargs) as outfile:
                    shutil.copyfileobj(cbuf, outfile)
        else:
            with self.tofile.open(key, **kwargs) as outfile:
                outfile.write(data)

    def close(self):
        if self.tofile is not None:
            self.tofile.close()
            self.tofile = None

import zipfile
import io
import typing
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_mean_std(arr):
    return f'Mean: {np.mean(np.array([*arr]), axis=0)}, Std: {np.std(np.array([*arr]), axis=0)}'

def rvec_2_euler(rvec):
    EULER = 'zyx'
    euler_rvec = R.from_rotvec(rvec).as_quat()
    return euler_rvec

def get_transformation(trans, rot):
    #breakpoint()
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

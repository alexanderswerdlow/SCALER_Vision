import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pickle 
from collections import defaultdict
from scipy.spatial import KDTree
from PIL import Image

import os
for file in os.listdir(f"data_files/captures"):
    if file.endswith(".npz"):
        print(file)
        loaded = np.load(f"data_files/captures/{file}", allow_pickle=True)
        frames = list(loaded.keys())
        for frame_key in frames:
            try:
                color_image, depth_image, _, _, _, _ = loaded[frame_key]
                Image.fromarray(color_image).save(f'data_files/all_data/rgb/{file.rstrip("npz")}_{frame_key}.png')
                Image.fromarray(depth_image).save(f'data_files/all_data/depth/{file.rstrip("npz")}_{frame_key}.png')
            except:
                break
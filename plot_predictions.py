import numpy as np
import matplotlib.pyplot as plt
import pickle 
from collections import defaultdict
from scipy.spatial import KDTree

# For getting rgb/depth image from captures
"""
loaded = np.load(f"data_files/capture.npz", allow_pickle=True)
frames = list(loaded.keys())
for frame_key in frames:
    try:
        color_image, depth_image, ir_image, intrinsic, trans, rot = loaded[frame_key]
        d435_data_list_all.append((color_image, depth_image, intrinsic))
        transformation_matrix_set.append(get_transformation(trans, rot) @ T265_to_D435_mat)
    except:
        break
"""

in_to_m = 0.0254
handhold_gt = np.array([[5, 0], [8, 4], [8, 9], [10, 12], [8, -4], [0, -5], [0, 5], [-6, -3], [-6, 3], [-17, -4], [-17, 2], [-14, 10]]).astype(np.float64)
handhold_gt *= in_to_m
handhold_tree = KDTree(handhold_gt)
handholds = defaultdict(list)

with open('data_files/generated/predictions.p', 'rb') as handle:
    all_predictions = pickle.load(handle)

plt.figure()
for idx, (center, axes, frame_rec) in enumerate(all_predictions):
    dist, idx = handhold_tree.query(center[:2])
    if dist < 0.05: # Ignore if more than 5cm away
        handholds[idx].append((center, axes, frame_rec))
        plt.scatter(*(center[:2] * 1 / in_to_m)) # Plot in inches
    else:
        print("More than 5cm away")

plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.savefig('data_files/plots/centroid.png')
plt.close()

#Numbered the holds from the upper right. 
axis_list = np.array([[84,30], [117,50], [107,30], [92,30], [87,30], [101,30], [150,30], [83,30], [93,30], [84,30], [85,30]])

def plot_error(ax, measurement):
    axis_errors = np.zeros([axis_list.shape[0], 2])
    axis_errors[:,0] = np.abs(np.max(measurement, axis=1) - axis_list[:,0])
    axis_errors[:,1] = np.abs(np.min(measurement, axis=1) - axis_list[:,1])

    width = 0.4
    r = np.arange(11)
    ax.bar(r, axis_errors[:,0],width=width, label = 'Maximum axis error')
    ax.bar(r+width, axis_errors[:,1],width=width, label = 'Minimum axis error')
    ax.set_xlabel('Bouldering hold ID')
    ax.set_title('Errors in Maximum/Minimum Axis Estimation for each Bouldering Hold')
    ax.set_ylabel('Error (mm)')
    ax.set_xticks(r + width/2,['0','1','2','3','4','5','6','7','8','9','10'])
    ax.legend()

np.random.seed(0)
measurement = np.random.uniform(low=25, high=120, size=(11,3))
fig, ax = plt.subplots(1)
plot_error(ax, measurement)
plt.savefig('data_files/plots/axis_error.png')
plt.close()



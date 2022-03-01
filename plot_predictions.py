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
handhold_gt = np.array([[10,12], [8,9], [8,4], [8,-4], [0,5], [0,-5], [-6,3], [-6,-3], [-14,10], [-17,2], [-17,4]]).astype(np.float64)

handhold_gt *= in_to_m
handhold_tree = KDTree(handhold_gt)
handholds = defaultdict(list)

with open('data_files/generated/2022_02_27-09_48_26_PM_predictions.p', 'rb') as handle:
    all_predictions = pickle.load(handle)

plt.figure()
for idx, (center, axes, frame_rec) in enumerate(all_predictions):
    dist, idx = handhold_tree.query(center[:2])
    if dist < 0.03: # Ignore if more than 5cm away
        handholds[idx].append((center, axes, frame_rec))
        plt.scatter(*(center[:2] * 1 / in_to_m)) # Plot in inches

plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.savefig('data_files/plots/centroid.png')
plt.close()

#Numbered the holds from the upper right. 
axis_gt = np.array([[84,30], [117,50], [107,30], [92,30], [87,30], [101,30], [150,30], [83,30], [93,30], [84,30], [85,30]])

def plot_axis_error(ax, axis_min, axis_max):
 
    axis_max_errors = np.abs(axis_gt[:,0] - axis_max)
    axis_min_errors = np.abs(axis_gt[:,1] - axis_min)

    axis_max_errors = axis_gt[:,0]- axis_max
    axis_min_errors = axis_gt[:,1]- axis_min

    # breakpoint()

    width = 0.4
    r = np.arange(len(axis_max))
    ax.bar(r, axis_max_errors,width=width, label = 'Minimum axis error')
    ax.bar(r+width, axis_min_errors,width=width, label = 'Maximum axis error')
    ax.set_xlabel('Bouldering hold ID')
    ax.set_title('Axis Size Estimation Errors')
    ax.set_ylabel('Error (mm)')
    ax.set_xticks(r + width/2, [str(x) for x in range(len(axis_max))])
    ax.legend()

def plot_center_error(ax, center_error_all):
    width = 0.4
    r = np.arange(len(axis_max))
    ax.bar(r, center_error_all.values() ,width=width, label = 'Centroid Error')
    #ax.bar(r+width, center_std_all.values() ,width=width, label = 'Centroid Standard Deviation')
    
    ax.set_xlabel('Bouldering hold ID')
    ax.set_title('Centroid Coordinate Estimation Errors')
    ax.set_ylabel('Error (mm)')
    ax.set_xticks(r + width/2,[str(x) for x in range(len(axis_max))])
    ax.legend()

#breakpoint()
center_error_all = {}
center_std_all = {}
axis_max = np.zeros(len(handholds))
axis_min = np.zeros(len(handholds))

center_std = np.zeros(len(handholds))
center_err_ = np.zeros(len(handholds))
axis_err_min = np.zeros(len(handholds))
axis_std_min = np.zeros(len(handholds))
axis_err_max = np.zeros(len(handholds))
axis_std_max = np.zeros(len(handholds))

axes = np.zeros([len(handholds),3])
all_center_errors = []
print(len(handholds))
for idx, handhold in handholds.items():
    if handhold: 
        center_arr = np.zeros([len(handhold), 3])
        axes_arr = np.zeros([len(handhold), 3])
        for j, handhold_frame in enumerate(handhold):
            center_arr[j] = handhold_frame[0]
            axes_arr[j] = handhold_frame[1]
            all_center_errors.append(handhold_frame[0][:2] - handhold_gt[idx])

        from PyNomaly import loop
        m = loop.LocalOutlierProbability(center_arr, extent=2, n_neighbors=20).fit()
        scores = m.local_outlier_probabilities
        for i in range(len(scores)):
            if scores[i] > 0.5:
                np.delete(axes_arr, i, 0)
                np.delete(center_arr, i, 0)
                    
        axes[idx] = np.median(axes_arr, axis=0)*1000*2
        center_median = np.median(center_arr, axis=0)
        
        axis_max[idx] = np.median(np.max(axes_arr, axis=1))*1000*2
        axis_min[idx] = np.median(np.min(axes_arr, axis=1))*1000*2
        # print('std dev axes', np.std(np.max(axes_arr*2, axis=1))*1000, np.std(np.max(axes_arr*2, axis=1))*1000)
        # print('centroid std dev', np.std(center_arr, axis=0) * 1000)

        #Euclidean distance error (m)
        center_error = np.linalg.norm(handhold_gt[idx] - center_median[:2])
        center_error_all[idx] = center_error*1000

        center_err_[idx] = np.linalg.norm(handhold_gt[idx] - center_median[:2])
        center_std[idx] = np.std(np.linalg.norm(handhold_gt[idx] - center_arr[:, :2], axis=1))
        
        axis_err_max[idx] = np.abs(axis_gt[idx,0] - np.median(np.max(axes_arr * 2 * 1000, axis=1)))
        axis_err_min[idx] = np.abs(axis_gt[idx,1] - np.median(np.min(axes_arr * 2 * 1000, axis=1)))

        axis_std_max[idx] = np.std(np.abs(axis_gt[idx,0] - np.max(axes_arr * 2 * 1000, axis=1)))
        axis_std_min[idx] = np.std(np.abs(axis_gt[idx,1] - np.min(axes_arr * 2 * 1000, axis=1)))
        
    else:
        center_error_all[idx] = np.inf
        # center_std_all[idx] = np.inf
        axis_max[idx] = np.inf
        axis_min[idx] = -np.inf


print(np.mean(center_err_, axis=0) * 1000)
print(np.mean(center_std, axis=0) * 1000)

print(np.mean(axis_err_max, axis=0))
print(np.mean(axis_err_min, axis=0))
print(np.mean(axis_std_max, axis=0))
print(np.mean(axis_std_min, axis=0))


from scipy.stats import norm
all_center_errors = np.array(all_center_errors).flatten()
plt.hist(all_center_errors, bins=20)
mu, std = norm.fit(all_center_errors)
print(mu*1000)
print(std*1000)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

plt.savefig('data_files/plots/center_error_dist.png')
exit()

fig, (ax1, ax2) = plt.subplots(1,2, constrained_layout=True) 
plot_center_error(ax1, center_error_all)
plot_axis_error(ax2, axis_min=axis_min, axis_max=axis_max)
# plt.show()
plt.savefig('data_files/plots/axis_error.png')
plt.close()
# np.random.seed(0)
# measurement = np.random.uniform(low=25, high=120, size=(11,3))
# fig, ax = plt.subplots(1)
# plot_error(ax, measurement)
# plt.savefig('data_files/plots/axis_error.png')
# plt.close()
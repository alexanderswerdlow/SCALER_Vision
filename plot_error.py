import numpy as np
import matplotlib.pyplot as plt

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
   
if __name__ == '__main__':
    np.random.seed(0)
    measurement = np.random.uniform(low=25, high=120, size=(11,3))
    fig, ax = plt.subplots(1)
    plot_error(ax, measurement)
    plt.show()

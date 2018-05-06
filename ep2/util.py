import math
import numpy as np
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix

def plot_histogram(labels):
    """
    Plot dataset histogram, counting the number of occurancies of each class in labels
    
    Doesn't require plt.show(), just call this function at the end of a cell.

    :param labels: labels of the dataset 
    :type labels: np.array
    """
    bins = len(np.unique(labels))
    data_hist = plt.hist(labels, 
                         bins=np.arange(bins+1) - 0.5, 
                         edgecolor='black')
    axes = plt.gca()  # Get Current Axes
    axes.set_ylim([0, len(labels)])
    plt.title("Histogram of {} images".format(len(labels)))
    plt.xticks(np.arange(bins+1), 
               np.unique(labels))
    plt.xlabel("Label")
    plt.ylabel("Frequency")
    for i in range(bins):
        plt.text(data_hist[1][i] + 0.1,
                 data_hist[0][i] + (data_hist[0][i] * 0.2),
                 str(int(data_hist[0][i])))

    plt.plot()

def plot_img_grid(data, 
                  labels, 
                  max_cols=1):
    """
    Plot image vectors as full sized images with their labels in a grid-like form (maximum of 20 images). 
    The number of columns is determined by max_cols, being the default a single column.
    
    Doesn't require plt.show(), just call this function at the end of a cell.
    
    :param data: data as vectors 
    :type data: np.array
    :param labels: labels of the dataset 
    :type labels: np.array
    :param max_cols: maximum number of columns in the grid (default=1)
    :type max_cols: int
    """
    assert len(labels) <= 20, "Too many images to display, limit is 20."
    max_rows = math.ceil(len(labels) / max_cols)
    for i in range(len(labels)):
        plt.subplot(max_rows, 
                    max_cols, 
                    i + 1)
        array_imshow(data[i],
                     labels[i], 
                     grid=False, 
                     fontsize=10)
    plt.tight_layout()
    
def array_imshow(img_array, 
                 label=None, 
                 shape=(28,28), 
                 grid=False, 
                 fontsize=18):
    """
    Plot a single image vector with its label. Since img_array is a vector, it's important
    to especify its original shape to resize it, for MNIST is height: 28 width: 28.  
    The plot can also show the image in a grid, if grid=True. Parameter fontsize is related to
    the size of the label font.
    
    Doesn't require plt.show(), just call this function at the end of a cell.
    
    :param img_array: flatten image (vector representation) 
    :type data: np.array
    :param label: label of the image (default=None)
    :type label: np.array
    :param shape: tuple with the original image dimensions (height, width, channels) (default=(28,28)) 
    :type shape: tuple of ints
    :param grid: show grid when plotting img_array
    :type grid: boolean
    :param fontsize: size of font when plotting the image label
    :type fontsize: int
    """
    img = array2img(img_array, 
                    shape=shape)
    plt.xlabel("Digit: {}".format(label), 
               fontsize=fontsize)
    plt.grid(grid)
    plt.imshow(img, 
               cmap='binary')

def array2img(img_array, 
              shape=(28,28)):
    """
    Given an image vector (img_array), returns a np.array with the image original dimensions (default=(28,28))
    
    :param img_array: flatten image (vector representation) 
    :type data: np.array
    :param shape: tuple with the original image dimensions (height, width, channels) (default=(28,28))
    :type shape: tuple of ints
    """
    return np.reshape(img_array, shape)

def get_samples(data, 
                labels, 
                size=3):
    idx_samples = np.random.randint(low=0, 
                                    high=len(labels),
                                    size=size)
    """
    Given the arrays data and labels, returns a tuple sample_data, sample_labels containing a number of elements specified in size parameter (default=3)
    sampled from data and labels. 
    The sampled dataset is a random process without replacement.
    
    :param data: data as vectors 
    :type data: np.array
    :param labels: labels of the dataset 
    :type labels: np.array
    :param size: number of elements in the sampled dataset
    :type size: int (default=3)
    """
    sample_data = np.array([data[idx] for idx in idx_samples])
    sample_labels = np.array([labels[idx] for idx in idx_samples])
    return sample_data, sample_labels

def plot_confusion_matrix_metrics(true_labels=None, 
                                  predicted_labels=None, 
                                  normalized=False):
    """
    Plot a confusion matrix given the known labels of the data (true_labels) and their corresponding predictions (predicted_labels).
    If normalized=True, the confusion matrix will bound its values in an interval between 0 and 1.
    
    Doesn't require plt.show(), just call this function at the end of a cell.
    
    :param true_labels: true values for labels
    :type true_labels: np.array
    :param predicted_labels: predicted label values
    :type predicted_labels: np.array
    :param normalized: bound the analysis in the interval [0, 1]
    :type normalized: boolean (default=False)
    """
    cm = ConfusionMatrix(true_labels, 
                         predicted_labels)
    cm.plot(cmap='GnBu', 
            normalized=normalized)
    ax = plt.gca()
    label_dict = {"True": 1, "False": 0}
    str_labels = [ 'Digit {}'.format(label_dict.get(i.get_text(), i.get_text())) for i in ax.get_xticklabels() ] 
    ax.set_xticklabels(str_labels, 
                       rotation=0, 
                       horizontalalignment='center')
    ax.set_yticklabels(str_labels)
    cm_array = cm.to_array()
    width, height = cm_array.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm_array[x][y]), 
                         xy=(y, x), 
                         horizontalalignment='center',
                         verticalalignment='center')
    plt.show()
    print(cm)
    print("===================================================")
    print("Evaluation metrics:")
    cm.print_stats()

import numpy as np
from codes.experiment_manager import *

import matplotlib.pyplot as plt
import numpy as np

def plot_curve(data, title='Curve Plot', xlabel='X-axis', ylabel='Y-axis'):
    """
    Plots a smooth curve from a list of data points.
    
    :param data: List of y-values.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    """
    x = np.linspace(0, len(data) - 1, len(data))
    y = np.array(data)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', linestyle='-', color='b', markersize=1, label='Data')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_curve_two(data1, data2, title='Curve Plot', xlabel='X-axis', ylabel='Y-axis', labels=('Data1', 'Data2')):
    """
    Plots smooth curves from two lists of data points.
    
    :param data1: List of y-values for the first dataset.
    :param data2: List of y-values for the second dataset.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param labels: Tuple containing labels for the two datasets.
    """
    x1 = np.linspace(0, len(data1) - 1, len(data1))
    y1 = np.array(data1)
    
    x2 = np.linspace(0, len(data2) - 1, len(data2))
    y2 = np.array(data2)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x1, y1, marker='o', linestyle='-', color='b', markersize=1, label=labels[0])
    plt.plot(x2, y2, marker='s', linestyle='--', color='r', markersize=1, label=labels[1])
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_matrix(matrix, title='Matrix Visualization'):
    """
    Visualizes a 2D matrix with values between [-1, 1] using a heatmap.
    
    :param matrix: 2D numpy array or list of lists with values in the range [-1, 1].
    :param title: Title of the plot.
    """
    plt.figure(figsize=(6, 5))
    cmap = plt.get_cmap('bwr')  # Blue-White-Red colormap to represent negative and positive values
    plt.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.show()
    
def plot_metrics_from_npz(file_path, keys, title, ylabel, markersize=4):
    """Generalized function to plot multiple metrics from an .npz file."""
    # Load data
    one = np.load(file_path)

    # Extract data for specified keys
    data = {key: one[key] for key in keys}

    # Find the minimum length among the data arrays
    min_length = min(len(arr) for arr in data.values())

    # Truncate arrays to the same length if needed
    for key in keys:
        if len(data[key]) > min_length:
            data[key] = data[key][-min_length:]  # Pop from the front to match lengths

    # Define x-axis (assuming same length for all metrics)
    epochs = np.arange(len(data[keys[0]]))

    # Define different line styles and markers for clarity
    styles = [
        ('-', 'o'),  # solid line with circle markers
        ('--', 's'), # dashed line with square markers
        (':', '^'),  # dotted line with triangle markers
        ('-.', 'x'), # dash-dot line with 'x' markers
        ('-', 'd')   # solid line with diamond markers
    ]

    plt.figure(figsize=(10, 6))

    for (key, (linestyle, marker)) in zip(keys, styles):
        plt.plot(epochs, data[key], linestyle=linestyle, marker=marker, 
                 alpha=0.8, linewidth=2, markersize=markersize, label=key.replace('_', ' ').title())

    # Labels and legend
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
    
def plot_validation_accuracies(file_path):
    """Plots validation accuracies from an .npz file."""
    keys = ['server_vali_acc', 'benign_mean_vali_acc', 
            'mali_trained_vali_acc', 'mali_normalized_vali_acc', 'mali_scaled_vali_acc']
    plot_metrics_from_npz(file_path, keys, "Validation Accuracies Over Epochs", "Accuracy")

def plot_gradient_norms(file_path):
    """Plots gradient norms from an .npz file."""
    keys = ['benign_grad_norm', 'mali_grad_norm']
    plot_metrics_from_npz(file_path, keys, "Gradient Norms Over Epochs", "Gradient Norm")

def plot_cosine_similarities(file_path):
    """Plots cosine similarities from an .npz file."""
    keys = ['attack_cos_budget','trained_cos', 'normalized_cos', 'scaled_cos', 'server_to_benign_cos']
    plot_metrics_from_npz(file_path, keys, "Cosine Dist Over Epochs", "Cosine Dist")

def plot_vali_acc(file_path):
    """Plots acc from an .npz file."""
    keys = ['server_val_test_accuracy']
    plot_metrics_from_npz(file_path, keys, "Accuracy Over Epochs", "Accuracy")
    
def plot_mali_selection(file_path):
    """Plots malicous client selection from an .npz file."""
    keys = ['select_percentage']
    plot_metrics_from_npz(file_path, keys, "Mali Client Selection % Over Epochs", "Selection%")    
    


def plot_metrics_with_two_y_axes(file_path, keys1, keys2, title, ylabel1, ylabel2, markersize=4): 
    """Plots two metrics with separate y-axes from an .npz file."""
    # Load data
    one = np.load(file_path)

    # Extract data for specified keys
    data1 = {key: one[key] for key in keys1}
    data2 = {key: one[key] for key in keys2}

    # Find the minimum length among the data arrays
    min_length = min(min(len(arr) for arr in data1.values()), min(len(arr) for arr in data2.values()))

    # Truncate arrays to the same length if needed
    for key in keys1:
        if len(data1[key]) > min_length:
            data1[key] = data1[key][-min_length:]  # Truncate from the front
    for key in keys2:
        if len(data2[key]) > min_length:
            data2[key] = data2[key][-min_length:]

    # Define x-axis based on the truncated length
    epochs = np.arange(min_length)

    # Define styles for plotting
    style1 = ('-', 'o')  # Line style and marker for first plot
    style2 = ('--', 's') # Line style and marker for second plot

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the first metric
    ax1.plot(epochs, data1[keys1[0]], linestyle=style1[0], marker=style1[1],
             alpha=0.8, linewidth=2, markersize=markersize, label=keys1[0].replace('_', ' ').title())
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel(ylabel1, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    # Plot the second metric with a shared x-axis but different y-axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, data2[keys2[0]], linestyle=style2[0], marker=style2[1],
             alpha=0.8, linewidth=2, markersize=markersize, color='tab:red', label=keys2[0].replace('_', ' ').title())
    ax2.set_ylabel(ylabel2, color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Title and show
    plt.title(title)
    fig.tight_layout()
    plt.show()

def plot_vali_acc_and_mali_selection(file_path):
    keys1 = ['server_val_test_accuracy']
    keys2 = ['select_percentage']
    plot_metrics_with_two_y_axes(file_path, keys1, keys2, 
                                 "Accuracy and Malicious Client Selection Over Epochs", 
                                 "Accuracy", "Selection %")

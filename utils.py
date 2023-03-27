from torch.cuda import is_available as cuda_available
from torch.backends.mps import is_available as mps_available

import matplotlib.pyplot as plt

def get_device():
    # Set GPU device
    if cuda_available():
        device = 'cuda'
    elif mps_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device


def plot_img(image):
    imgplot = plt.imshow(image)
    plt.axis('off')
    imgplot.axes.get_xaxis().set_visible(False)
    imgplot.axes.get_yaxis().set_visible(False)
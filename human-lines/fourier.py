import glob

import numpy as np

def fourier(img: np.ndarray):
    """Fourier Based Decomposition of the image
    
    Args:
        frame (np.ndarray) image to be analyzed
    
    Returns:
        principal frequencies of the image
    """

    fft = np.fft.fft2(img) 



if __name__ == "__main__":

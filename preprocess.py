import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.fft import fft2, fftshift
from scipy.signal import correlate2d
import time
import os




def extract_noise_residual(image, sigma=1.0):
    """
    Extract noise residuals from an image using Gaussian denoising.

    Parameters:
    - image: Input image as a 2D NumPy array (grayscale) or 3D NumPy array (color).
    - sigma: Standard deviation for the Gaussian filter.

    Returns:
    - residual: Noise residuals as a 2D NumPy array (grayscale) or 3D NumPy array (color).
    """
 
    residual = np.zeros_like(image)
    for channel in range(3):
        denoised_channel = gaussian_filter(image[:, :, channel], sigma=sigma)
        residual[:, :, channel] = image[:, :, channel] - denoised_channel

    # Normalize the residual for visualization
    residual = residual - np.min(residual)
    residual = residual / np.max(residual)

    return residual

def compute_power_spectrum(residual):
    """
    Compute the power spectrum of the noise residual.

    Parameters:
    - residual: Noise residual as a 3D NumPy array (H x W x 3).

    Returns:
    - power_spectrum: The average power spectrum across all color channels.
    """
    power_spectra = []
    for channel in range(3):
        # Compute the 2D Fourier Transform and shift the zero-frequency component to the center
        f_transform = fft2(residual[:, :, channel])
        f_shifted = fftshift(f_transform)

        # Calculate the power spectrum
        power_spectrum = np.abs(f_shifted) ** 2
        power_spectra.append(power_spectrum)
    
    # Average the power spectra across all channels
    average_power_spectrum = np.mean(power_spectra, axis=0)
    return average_power_spectrum

def compute_autocorrelation(residual):
    """
    Compute the autocorrelation of the noise residual.

    Parameters:
    - residual: Noise residual as a 3D NumPy array (H x W x 3).

    Returns:
    - autocorrelation: The average autocorrelation function across all color channels.
    """
    autocorrelations = []
    for channel in range(3):
        # Calculate 2D autocorrelation using correlate2d
        autocorr = correlate2d(residual[:, :, channel], residual[:, :, channel], mode='same')
        autocorrelations.append(autocorr)
    
    # Average the autocorrelations across all channels
    average_autocorrelation = np.mean(autocorrelations, axis=0)
    
    # Normalize the autocorrelation for visualization
    average_autocorrelation -= np.min(average_autocorrelation)
    average_autocorrelation /= np.max(average_autocorrelation)
    
    return average_autocorrelation


def sparse_sample_image(image, stride=2):
    """
    Perform sparse sampling of an image by taking every nth pixel based on the stride.
    
    Parameters:
    - image: 3D NumPy array (H x W x 3).
    - stride: Step size for sampling (default is 2).

    Returns:
    - sampled_image: The sparsely sampled image.
    """
    sampled_image = image[::stride, ::stride, :]
    return sampled_image


def crop_to_256x256(image):
    """
    Crop an image to 256x256. Assumes image is at least 256x256.
    """
    h, w, _ = image.shape
    start_x = w // 2 - 128
    start_y = h // 2 - 128
    return image[start_y:start_y + 256, start_x:start_x + 256]

def process_single_image(image_path, label, sigma=1.0):
    """
    Process a single image to extract features and label.
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            return None  # Skip invalid images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        image = sparse_sample_image(crop_to_256x256(image))
        
        # Extract noise residuals
        residual = extract_noise_residual(image, sigma=sigma)

        # Extract features
        feature_vector = np.concatenate([compute_autocorrelation(residual), compute_power_spectrum(residual)]).flatten()
        print("Processed image: " + image_path)
        return feature_vector, label
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def process_images_from_dir(directory, label, sigma=1.0, max_workers=4):
    """
    Process images from a directory (including subdirectories) to extract features.
    Utilizes multiprocessing for faster processing.
    """
    features = []
    labels = []

    # Get list of all image paths
    image_paths = [os.path.join(root, filename)
                   for root, _, files in os.walk(directory)
                   for filename in files]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks for each image to be processed
        futures = [executor.submit(process_single_image, image_path, label, sigma) for image_path in image_paths]

        for future in as_completed(futures):
            result = future.result()
            if result:
                feature_vector, lbl = result
                features.append(feature_vector)
                labels.append(lbl)

    return features, labels

def process_dataset(real_dir, fake_dir, max_workers=4):
    print("loading datasets: ", fake_dir, real_dir)
    start_time = time.time()
    real_feat, real_label = process_images_from_dir(real_dir, label=0, max_workers=max_workers)
    fake_feat, fake_label = process_images_from_dir(fake_dir, label=1, max_workers=max_workers)

    features = np.concatenate((real_feat, fake_feat), axis=0)
    labels = np.concatenate((real_label, fake_label), axis=0)
    print("data loaded in ", time.time() - start_time)
    return features, labels

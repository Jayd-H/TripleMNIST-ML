# Jayden Holdsworth 2024
# Task 1: Visualisation
# Algorithm to load and display sample images from the triple MNIST dataset

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
import random

# load random samples 
def load_sample_images(data_path, num_folders=5, images_per_folder=3):
    images = []
    labels = []
    
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    selected_folders = random.sample(folders, min(num_folders, len(folders)))
    
    for folder in selected_folders:
        folder_path = os.path.join(data_path, folder)
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        
        selected_files = random.sample(image_files, min(images_per_folder, len(image_files)))
        
        for img_file in selected_files:
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path)
            images.append(np.array(img))
            labels.append(folder)
    
    return images, labels

# display the samples loaded
def show_examples(images, labels):
    num_images = len(images)
    cols = 3
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
    axes = axes.flatten() if rows > 1 else [axes]
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f'Digits: {label}')
        axes[idx].axis('off')
    
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    
# displays non-zero pixel intensities 
def pixel_intensities(data_path):
    images, labels = load_sample_images(data_path, num_folders=10, images_per_folder=1)
    all_pixels = np.concatenate([img.flatten() for img in images])
    
    non_zero_pixels = all_pixels[all_pixels > 0]
    
    fig = plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(non_zero_pixels, bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Non-Zero Pixel Intensities')
    plt.xlabel('Pixel Value (1-255)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.vstack(images[:3]), cmap='gray')
    plt.title('Example Images')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_path = "data/train"
    
    print("Loading and displaying sample images...")
    images, labels = load_sample_images(data_path)
    show_examples(images, labels)
    
    print(f"\nBasic Image Information:")
    print(f"Image dimensions: {images[0].shape}")
    print(f"Pixel value range: {images[0].min()} to {images[0].max()}")
    
    print("\nAnalyzing pixel intensities and writing styles...")
    pixel_intensities(data_path)
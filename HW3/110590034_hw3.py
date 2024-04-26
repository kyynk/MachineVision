'''
Modules import
'''
import numpy as np
import cv2

def rgb_to_gray(image):
    '''
    Convert color image to grayscale image
    '''
    gray_image = np.dot(image[..., :3], [0.3, 0.59, 0.11])
    return gray_image.astype(np.uint8)

def to_binary(image, threshold):
    '''
    Convert image to binary image
    '''
    binary_image = np.where(image > threshold, 0, 255)
    return binary_image.astype(np.uint8)

def distance_transform_8(image):
    height, width = image.shape
    distance_map = np.zeros_like(image, dtype=np.uint16)

    for i in range(height):
        for j in range(width):
            if image[i, j] != 0:
                distance_map[i, j] = 1
            else:
                distance_map[i, j] = 0

    for m in range(1, max(height, width)):
        for i in range(height):
            for j in range(width):
                if image[i, j] == 0:
                    continue
                neighbors = []
                for u in range(max(0, i-1), min(height, i+2)):
                    for v in range(max(0, j-1), min(width, j+2)):
                        if distance_map[u, v] != 0:
                            neighbors.append(distance_map[u, v])
                if neighbors:
                    distance_map[i, j] = min(neighbors) + 1

    return distance_map
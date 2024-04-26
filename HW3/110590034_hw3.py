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

def gray_to_binary(image, threshold):
    '''
    Convert image to binary image
    '''
    binary_image = np.where(image > threshold, 0, 255)
    return binary_image.astype(np.uint8)

def distance_transform_8(image):
    '''
    Distance transform with 8 neighbors
    '''
    height, width = image.shape
    distance_map = np.zeros_like(image, dtype=np.uint16)
    distance_map = np.where(image == 255, 1, 0)

    # print(distance_map)
    top_left_neighbors = [(-1, 0), (-1, -1), (-1, 1), (0, -1)]

    for i in range(height):
        for j in range(width):
            if image[i, j] == 0:
                continue
            neighbors = []
            for n_y, n_x in top_left_neighbors:
                if 0 <= (i + n_y) < height and 0 <= (j + n_x) < width:
                    neighbors.append(distance_map[i + n_y, j + n_x])
            # print(len(neighbors))
            if neighbors:
                distance_map[i, j] = min(neighbors) + 1
    
    # print(distance_map)
    bottom_right_neighbors = [(1, 0), (1, 1), (1, -1), (0, 1)]

    for i in range(height - 1, -1, -1):
        for j in range(width - 1, -1, -1):
            if image[i, j] == 0:
                continue
            neighbors = []
            neighbors.append(distance_map[i, j])
            for n_y, n_x in bottom_right_neighbors:
                if 0 <= (i + n_y) < height and 0 <= (j + n_x) < width:
                    neighbors.append(distance_map[i + n_y, j + n_x])
            # print(len(neighbors), neighbors)
            if neighbors:
                distance_map[i, j] = min(neighbors) + 1

    return distance_map

def colorize_distance_transform(distance_image):
    '''
    Colorize distance transform image
    '''
    max_value = np.max(distance_image)
    colors_distance_image = (distance_image / max_value * 255).astype(np.uint8)  # 將數值映射到 [0, 255] 的範圍
    return colors_distance_image

def image(number, threshold):
    '''
    For img{number}.png
    '''
    image = cv2.imread(f'images/img{number}.jpg')
    gray_image = rgb_to_gray(image)
    binary_image = gray_to_binary(gray_image, threshold)
    distance_image = distance_transform_8(binary_image)
    colors_distance_image = colorize_distance_transform(distance_image)

    cv2.imshow('Original Image', image)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.imshow('Binary Image', binary_image)
    cv2.imshow('Colors Distance Transform Image', colors_distance_image)
    cv2.imwrite(f'results/img{number}_q1-1.jpg', colors_distance_image)
    # cv2.imwrite(f'results/img{number}_q1-2.jpg', )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image(number=1, threshold=200)
    image(number=2, threshold=200)
    image(number=3, threshold=200)
    image(number=4, threshold=200)

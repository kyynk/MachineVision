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

def image(number, threshold):
    '''
    For img{number}.png
    '''
    image = cv2.imread(f'images/img{number}.jpg')
    gray_image = rgb_to_gray(image)
    binary_image = gray_to_binary(gray_image, threshold)

    cv2.imshow('Original Image', image)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.imshow('Binary Image', binary_image)
    # cv2.imwrite(f'results/img{number}_q1-1.jpg', colors_distance_image_4)
    # cv2.imwrite(f'results/img{number}_q1.jpg', colors_distance_image_8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image(number=1, threshold=100)
    image(number=2, threshold=100)
    image(number=3, threshold=100)

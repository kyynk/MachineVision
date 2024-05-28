'''
Modules import
'''
import numpy as np
import cv2

def get_n_kernel(image, row, col, kernel_size):
    '''
    Get n kernel
    '''
    height = image.shape[0]
    width = image.shape[1]
    kernel = np.zeros((kernel_size, kernel_size)).astype(np.float64)
    for i in range(-kernel_size//2, kernel_size//2+1):
        for j in range(-kernel_size//2, kernel_size//2+1):
            if 0 <= row+i < height and 0 <= col+j < width:
                kernel[i+kernel_size//2][j+kernel_size//2] = image[row+i][col+j][0]

    return kernel

def mean_filter(image, kernel_size):
    '''
    Mean filter
    '''
    height = image.shape[0]
    width = image.shape[1]
    image = image.astype(np.float64)
    mean_filtered_image = np.zeros((height, width)).astype(np.float64)
    # for row in range(height):
    #     for col in range(width):
    #         if image[row][col][0] != image[row][col][1] or \
    #            image[row][col][2] != image[row][col][1] or \
    #            image[row][col][2] != image[row][col][0]:
    #             print('diff')

    # print(get_n_kernel(image, 0, 0, kernel_size))

    for row in range(height):
        for col in range(width):
            kernel = get_n_kernel(image, row, col, kernel_size)
            mean_filtered_image[row][col] = np.sum(kernel) / kernel_size**2
    return mean_filtered_image

def median_filter(image, kernel_size):
    '''
    Mean filter
    '''
    height = image.shape[0]
    width = image.shape[1]
    image = image.astype(np.float64)
    median_filtered_image = np.zeros((height, width)).astype(np.float64)
    
    # print(get_n_kernel(image, 0, 0, kernel_size))
    # print(np.median(get_n_kernel(image, 0, 0, kernel_size)))

    for row in range(height):
        for col in range(width):
            kernel = get_n_kernel(image, row, col, kernel_size)
            median_filtered_image[row][col] = np.median(kernel)
    return median_filtered_image

def gaussian_kernel(kernel_size, sigma):
    '''
    Gaussian kernel
    '''
    kernel = np.zeros((kernel_size, kernel_size)).astype(np.float64)
    for x in range(-kernel_size//2, kernel_size//2+1):
        for y in range(-kernel_size//2, kernel_size//2+1):
            kernel[x+kernel_size//2][y+kernel_size//2] = 1 / (2 * np.pi * sigma**2) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel

def gaussian_filter(image, kernel_size, sigma):
    '''
    Gaussian filter
    '''
    height = image.shape[0]
    width = image.shape[1]
    image = image.astype(np.float64)
    gaussian_filtered_image = np.zeros((height, width)).astype(np.float64)
    gaussian = gaussian_kernel(kernel_size, sigma)
    gaussian = gaussian / np.sum(gaussian)
    for row in range(height):
        for col in range(width):
            kernel = get_n_kernel(image, row, col, kernel_size)
            gaussian_filtered_image[row][col] = np.sum(kernel * gaussian)
    return gaussian_filtered_image

def custom_filter(image, kernel_size, sigma, filter_type):
    '''
    Custom filter
    1. mean filter + median filter
    2. mean filter + gaussian filter
    3. median filter + gaussian filter
    4. mean filter + median filter + gaussian filter
    '''
    height = image.shape[0]
    width = image.shape[1]
    image = image.astype(np.float64)
    custom_filtered_image = np.zeros((height, width)).astype(np.float64)
    gaussian = gaussian_kernel(kernel_size, sigma)
    gaussian = gaussian / np.sum(gaussian)
    for row in range(height):
        for col in range(width):
            kernel = get_n_kernel(image, row, col, kernel_size)
            if filter_type == 1:
                custom_filtered_image[row][col] = (np.sum(kernel) / kernel_size**2 + np.median(kernel)) / 2
            elif filter_type == 2:
                custom_filtered_image[row][col] = (np.sum(kernel) / kernel_size**2 + np.sum(kernel * gaussian)) / 2
            elif filter_type == 3:
                custom_filtered_image[row][col] = (np.median(kernel) + np.sum(kernel * gaussian)) / 2
            elif filter_type == 4:
                custom_filtered_image[row][col] = (np.sum(kernel) / kernel_size**2 + np.median(kernel) + np.sum(kernel * gaussian)) / 3

    return custom_filtered_image

def image(number, sigma):
    '''
    For img{number}.png
    '''
    origin_image = cv2.imread(f'images/img{number}.jpg')

    mean_filtered_image_3 = mean_filter(origin_image, 3)
    cv2.imwrite(f'results/img{number}_q1_3.jpg', mean_filtered_image_3)
    mean_3 = cv2.imread(f'results/img{number}_q1_3.jpg')

    mean_filtered_image_7 = mean_filter(origin_image, 7)
    cv2.imwrite(f'results/img{number}_q1_7.jpg', mean_filtered_image_7)
    mean_7 = cv2.imread(f'results/img{number}_q1_7.jpg')

    median_filtered_image_3 = median_filter(origin_image, 3)
    cv2.imwrite(f'results/img{number}_q2_3.jpg', median_filtered_image_3)
    median_3 = cv2.imread(f'results/img{number}_q2_3.jpg')

    median_filtered_image_7 = median_filter(origin_image, 7)
    cv2.imwrite(f'results/img{number}_q2_7.jpg', median_filtered_image_7)
    median_7 = cv2.imread(f'results/img{number}_q2_7.jpg')

    gaussian_filtered_image = gaussian_filter(origin_image, 5, sigma)
    cv2.imwrite(f'results/img{number}_q3.jpg', gaussian_filtered_image)
    gaussian = cv2.imread(f'results/img{number}_q3.jpg')

    for i in range(1, 5):
        custom_filtered_image = custom_filter(origin_image, 5, sigma, i)
        cv2.imwrite(f'results/img{number}_q4_{i}.jpg', custom_filtered_image)

    cv2.imshow('Origin Image', origin_image)
    cv2.imshow('Mean Filtered Image (3x3)', mean_3)
    cv2.imshow('Mean Filtered Image (7x7)', mean_7)
    cv2.imshow('Median Filtered Image (3x3)', median_3)
    cv2.imshow('Median Filtered Image (7x7)', median_7)
    cv2.imshow('Gaussian Filtered Image', gaussian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image(number=1, sigma=1)
    image(number=2, sigma=1.4)
    image(number=3, sigma=1.2)

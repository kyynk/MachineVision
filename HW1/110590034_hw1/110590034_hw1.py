'''
Modules import
'''
import numpy as np
import cv2

def rgb_to_gray(image):
    '''
    1-1. Convert color image to grayscale image
    '''
    gray_image = np.dot(image[..., :3], [0.3, 0.59, 0.11])
    return gray_image.astype(np.uint8)

def gray_to_binary(gray_image, threshold):
    '''
    1-2. Convert grayscale image to binary image
    '''
    binary_image = np.where(gray_image > threshold, 255, 0)
    return binary_image.astype(np.uint8)

# color_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def find_closest_color(pixel):
    '''
    Convert each pixel to its closest color in the custom colormap
    '''
    custom_colormap = np.array([
        [241, 199, 106],# Yellow
        [240, 230, 140],# Khaki
        [0, 0, 255],    # Blue
        [0, 255, 255],  # Cyan
        [222, 184, 135],# BurlyWood
        [255, 255, 224],# LightYellow
        [128, 0, 0],    # Maroon
        [0, 128, 0],    # Green
        [0, 0, 128],    # Navy
        [0, 128, 128],  # Teal
        [198, 110, 72], # Brown
        [128, 128, 0],  # Olive
        [0, 0, 0],      # Black
        [128, 128, 128],# Gray
        [192, 192, 192],# Silver
        [255, 255, 255] # White
    ])
    distances = np.linalg.norm(custom_colormap - pixel, axis=1)
    # color_count[np.argmin(distances)] += 1
    return np.argmin(distances)

def color_to_index(image):
    '''
    1-3. Convert color image to index-color image
    '''
    custom_colormap = np.array([
        [241, 199, 106],# Yellow
        [240, 230, 140],# Khaki
        [0, 0, 255],    # Blue
        [0, 255, 255],  # Cyan
        [222, 184, 135],# BurlyWood
        [255, 255, 224],# LightYellow
        [128, 0, 0],    # Maroon
        [0, 128, 0],    # Green
        [0, 0, 128],    # Navy
        [0, 128, 128],  # Teal
        [198, 110, 72], # Brown
        [128, 128, 0],  # Olive
        [0, 0, 0],      # Black
        [128, 128, 128],# Gray
        [192, 192, 192],# Silver
        [255, 255, 255] # White
    ])
    index_image = np.zeros_like(image, dtype=np.uint8)
    # print(image)
    # print(index_image)
    # print(image.shape[0])
    # print(image.shape[1])
    # print(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # print('a= ', image[i, j])
            index_image[i, j] = custom_colormap[find_closest_color(image[i, j])]
            # print(index_image[i, j])
    # for i in range(image.shape[1]):
    #     for j in range(image.shape[0]):
    #         index_image[i, j] = custom_colormap[index_image[i, j]]
    # print(image)
    # print(index_image)
    return index_image

def resize_no_interpolation(image, scale):
    '''
    2-1. Define function for resizing without interpolation
    '''
    height, width = image.shape[:2]

    new_height = int(height * scale)
    new_width = int(width * scale)

    resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for new_i in range(new_height):
        for new_j in range(new_width):
            original_i = int(new_i / scale)
            original_j = int(new_j / scale)
            resized_image[new_i, new_j] = image[original_i, original_j]

    return resized_image

def bilinear_interpolation(image, scale):
    '''
    2-2. Define function for bilinear interpolation
    '''
    height, width = image.shape[:2]

    new_height = int(height * scale)
    new_width = int(width * scale)

    resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for new_i in range(new_height):
        for new_j in range(new_width):
            original_i = new_i / scale
            original_j = new_j / scale

            # Get the surrounding four pixels in the original image
            top_left = image[int(original_i), int(original_j)]
            top_right = image[int(original_i), min(int(original_j) + 1, width - 1)]
            bottom_left = image[min(int(original_i) + 1, height - 1), int(original_j)]
            bottom_right = image[min(int(original_i) + 1, height - 1), min(int(original_j) + 1, width - 1)]

            # Calculate the weights for interpolation
            dx = original_j - int(original_j)
            dy = original_i - int(original_i)

            # Perform bilinear interpolation
            top_interpolation = (1 - dx) * top_left + dx * top_right
            bottom_interpolation = (1 - dx) * bottom_left + dx * bottom_right
            resized_image[new_i, new_j] = (1 - dy) * top_interpolation + dy * bottom_interpolation

    return resized_image

def img1():
    '''
    For img1.png
    '''
    image = cv2.imread('images/img1.png')
    # print(image)

    gray_image = rgb_to_gray(image)
    threshold = 100
    binary_image = gray_to_binary(gray_image, threshold)
    index_image = color_to_index(image)
    resized_image_no_interpolation_half = resize_no_interpolation(image, 0.5)
    resized_image_no_interpolation_double = resize_no_interpolation(image, 2)
    resized_image_bilinear_half = bilinear_interpolation(image, 0.5)
    resized_image_bilinear_double = bilinear_interpolation(image, 2)

    cv2.imshow('Original Image', image)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.imshow('Binary Image', binary_image)
    cv2.imshow('Index-Color Image', index_image)
    cv2.imshow('Resized without interpolation - 1/2', resized_image_no_interpolation_half)
    cv2.imshow('Resized without interpolation - 2 times', resized_image_no_interpolation_double)
    cv2.imshow('Resized with bilinear interpolation - 1/2', resized_image_bilinear_half)
    cv2.imshow('Resized with bilinear interpolation - 2 times', resized_image_bilinear_double)
    cv2.imwrite('results/img1_q1-1.png', gray_image)
    cv2.imwrite('results/img1_q1-2.png', binary_image)
    cv2.imwrite('results/img1_q1-3.png', index_image)
    cv2.imwrite('results/img1_q2-1-half.png', resized_image_no_interpolation_half)
    cv2.imwrite('results/img1_q2-1-double.png', resized_image_no_interpolation_double)
    cv2.imwrite('results/img1_q2-2-half.png', resized_image_bilinear_half)
    cv2.imwrite('results/img1_q2-2-double.png', resized_image_bilinear_double)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def img2():
    '''
    For img2.png
    '''
    image = cv2.imread('images/img2.png')
    # print(image)

    gray_image = rgb_to_gray(image)
    threshold = 155
    binary_image = gray_to_binary(gray_image, threshold)
    index_image = color_to_index(image)
    resized_image_no_interpolation_half = resize_no_interpolation(image, 0.5)
    resized_image_no_interpolation_double = resize_no_interpolation(image, 2)
    resized_image_bilinear_half = bilinear_interpolation(image, 0.5)
    resized_image_bilinear_double = bilinear_interpolation(image, 2)

    cv2.imshow('Original Image', image)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.imshow('Binary Image', binary_image)
    cv2.imshow('Index-Color Image', index_image)
    cv2.imshow('Resized without interpolation - 1/2', resized_image_no_interpolation_half)
    cv2.imshow('Resized without interpolation - 2 times', resized_image_no_interpolation_double)
    cv2.imshow('Resized with bilinear interpolation - 1/2', resized_image_bilinear_half)
    cv2.imshow('Resized with bilinear interpolation - 2 times', resized_image_bilinear_double)
    cv2.imwrite('results/img2_q1-1.png', gray_image)
    cv2.imwrite('results/img2_q1-2.png', binary_image)
    cv2.imwrite('results/img2_q1-3.png', index_image)
    cv2.imwrite('results/img2_q2-1-half.png', resized_image_no_interpolation_half)
    cv2.imwrite('results/img2_q2-1-double.png', resized_image_no_interpolation_double)
    cv2.imwrite('results/img2_q2-2-half.png', resized_image_bilinear_half)
    cv2.imwrite('results/img2_q2-2-double.png', resized_image_bilinear_double)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def img3():
    '''
    For img3.png
    '''
    image = cv2.imread('images/img3.png')
    # print(image)

    gray_image = rgb_to_gray(image)
    threshold = 123
    binary_image = gray_to_binary(gray_image, threshold)
    index_image = color_to_index(image)
    resized_image_no_interpolation_half = resize_no_interpolation(image, 0.5)
    resized_image_no_interpolation_double = resize_no_interpolation(image, 2)
    resized_image_bilinear_half = bilinear_interpolation(image, 0.5)
    resized_image_bilinear_double = bilinear_interpolation(image, 2)

    cv2.imshow('Original Image', image)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.imshow('Binary Image', binary_image)
    cv2.imshow('Index-Color Image', index_image)
    cv2.imshow('Resized without interpolation - 1/2', resized_image_no_interpolation_half)
    cv2.imshow('Resized without interpolation - 2 times', resized_image_no_interpolation_double)
    cv2.imshow('Resized with bilinear interpolation - 1/2', resized_image_bilinear_half)
    cv2.imshow('Resized with bilinear interpolation - 2 times', resized_image_bilinear_double)
    cv2.imwrite('results/img3_q1-1.png', gray_image)
    cv2.imwrite('results/img3_q1-2.png', binary_image)
    cv2.imwrite('results/img3_q1-3.png', index_image)
    cv2.imwrite('results/img3_q2-1-half.png', resized_image_no_interpolation_half)
    cv2.imwrite('results/img3_q2-1-double.png', resized_image_no_interpolation_double)
    cv2.imwrite('results/img3_q2-2-half.png', resized_image_bilinear_half)
    cv2.imwrite('results/img3_q2-2-double.png', resized_image_bilinear_double)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img1()
    img2()
    img3()
    # print(color_count)

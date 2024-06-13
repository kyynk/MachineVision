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
    return gray_image.astype(np.float64)

def get_n_kernel(image, row, col, kernel_size):
    '''
    Get n kernel
    '''
    height = image.shape[0]
    width = image.shape[1]
    kernel = np.zeros((kernel_size, kernel_size)).astype(np.float64)
    for i in range(-kernel_size//2, kernel_size//2+1):
        for j in range(-kernel_size//2, kernel_size//2+1):
            # if 0 <= row+i < height and 0 <= col+j < width:
            #     kernel[i+kernel_size//2][j+kernel_size//2] = image[row+i][col+j]
            new_row = min(max(row + i, 0), height - 1)
            new_col = min(max(col + j, 0), width - 1)
            kernel[i + kernel_size//2][j + kernel_size//2] = image[new_row][new_col]
    return kernel

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

def sobel_filter(image):
    '''
    Sobel filter
    '''
    height = image.shape[0]
    width = image.shape[1]
    image = image.astype(np.float64)
    G = np.zeros((height, width)).astype(np.float64)
    theta = np.zeros((height, width)).astype(np.float64)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # kernel = get_n_kernel(image, 0, 0, 3)
    # print(np.hypot(np.sum(kernel * sobel_x), np.sum(kernel * sobel_y)))
    # print(np.sqrt(np.sum(kernel * sobel_x)**2 + np.sum(kernel * sobel_y)**2))
    for row in range(height):
        for col in range(width):
            kernel = get_n_kernel(image, row, col, 3)
            Gx = np.sum(kernel * sobel_x)
            Gy = np.sum(kernel * sobel_y)
            G[row][col] = np.hypot(Gx, Gy)
            theta[row][col] = np.arctan2(Gy, Gx)
    # G = G / G.max() * 255
    return G, theta

def non_maximum_suppression(image, theta):
    '''
    Non-maximum suppression
    '''
    height = image.shape[0]
    width = image.shape[1]
    suppressed_image = np.zeros((height, width)).astype(np.float64)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for row in range(1, height-1):
        for col in range(1, width-1):
            pixel_q = 255
            pixel_r = 255
            # angle 0
            if (0 <= angle[row][col] < 22.5) or (157.5 <= angle[row][col] <= 180):
                pixel_q = image[row][col-1]
                pixel_r = image[row][col+1]
            # angle 45
            elif (22.5 <= angle[row][col] < 67.5):
                pixel_q = image[row-1][col+1]
                pixel_r = image[row+1][col-1]
            # angle 90
            elif (67.5 <= angle[row][col] < 112.5):
                pixel_q = image[row-1][col]
                pixel_r = image[row+1][col]
            # angle 135
            elif (112.5 <= angle[row][col] < 157.5):
                pixel_q = image[row-1][col-1]
                pixel_r = image[row+1][col+1]
            
            if image[row][col] >= pixel_q and image[row][col] >= pixel_r:
                suppressed_image[row][col] = image[row][col]
            else:
                suppressed_image[row][col] = 0

    return suppressed_image

def double_threshold(image, low_threshold_ratio, high_threshold_ratio):
    '''
    Double threshold with dynamic thresholding
    '''
    high_threshold = image.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio
    height = image.shape[0]
    width = image.shape[1]
    double_threshold_image = np.zeros((height, width)).astype(np.float64)
    
    weak = 127.5
    strong = 255

    double_threshold_image[image >= high_threshold] = strong
    double_threshold_image[image < low_threshold] = 0
    double_threshold_image[(image >= low_threshold) & (image < high_threshold)] = weak

    return double_threshold_image, weak, strong

def edge_tracking(image, weak, strong):
    '''
    Edge tracking by Hysteresis
    '''
    height = image.shape[0]
    width = image.shape[1]
    for row in range(1, height-1):
        for col in range(1, width-1):
            if image[row][col] == weak:
                kernel = get_n_kernel(image, row, col, 3)
                if strong in kernel:
                    image[row][col] = strong
                else:
                    image[row][col] = 0
    return image

def image(number, gaussian_kernel_size, sigma, double_threshold_low, double_threshold_high):
    '''
    For img{number}.png
    '''
    if number == 1:
        origin_image = cv2.imread(f'images/img{number}.jpeg')
    else:
        origin_image = cv2.imread(f'images/img{number}.jpg')

    gray_image = rgb_to_gray(origin_image)
    cv2.imwrite(f'results/img{number}_1.jpg', gray_image)

    gaussian_filtered_image = gaussian_filter(gray_image, gaussian_kernel_size, sigma)
    cv2.imwrite(f'results/img{number}_2.jpg', gaussian_filtered_image)

    magnitude, slope = sobel_filter(gaussian_filtered_image)
    cv2.imwrite(f'results/img{number}_2_mag.jpg', magnitude)
    print(magnitude.max(), magnitude.min())

    suppressed_image = non_maximum_suppression(magnitude, slope)
    cv2.imwrite(f'results/img{number}_3.jpg', suppressed_image)

    double_threshold_image, weak, strong = double_threshold(suppressed_image, double_threshold_low, double_threshold_high)
    cv2.imwrite(f'results/img{number}_4.jpg', double_threshold_image)

    edge_image = edge_tracking(double_threshold_image, weak, strong)
    cv2.imwrite(f'results/img{number}_sobel.jpg', edge_image)

if __name__ == '__main__':
    image(number=1, gaussian_kernel_size=5, sigma=1.7, double_threshold_low=0.05, double_threshold_high=0.5)
    image(number=2, gaussian_kernel_size=5, sigma=1.7, double_threshold_low=0.15, double_threshold_high=0.3)
    image(number=3, gaussian_kernel_size=3, sigma=1.3, double_threshold_low=0.18, double_threshold_high=0.2)
